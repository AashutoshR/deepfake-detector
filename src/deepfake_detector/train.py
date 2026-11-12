"""Training loop: bf16, gradient checkpointing, accumulation, resumable runs (FR-T1 to FR-T9).

Every run is defined by the three YAML configs plus command-line overrides and the git commit,
all saved in the run folder (FR-T1). The best checkpoint is picked on FF++ validation video AUC
only; test sets are never touched here (FR-T6).

Examples:
    python -m deepfake_detector.train
    python -m deepfake_detector.train train.leave_out_method=Deepfakes train.run_name=lomo_df
    python -m deepfake_detector.train model.arch=xception train.run_name=xception
    python -m deepfake_detector.train --resume runs/<run>/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from deepfake_detector.data.augment import TrainTransform
from deepfake_detector.data.datasets import FaceCropDataset, load_manifest
from deepfake_detector.data.extract import sample_indices
from deepfake_detector.evaluate import predict_frames
from deepfake_detector.metrics import auc_per_method, summarize
from deepfake_detector.models import build_model
from deepfake_detector.models.common import IGNORE_INDEX, count_parameters, method_target

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- config and env


def load_config(
    data: str | Path = "configs/data.yaml",
    model: str | Path = "configs/model.yaml",
    train: str | Path = "configs/train.yaml",
    overrides: Sequence[str] = (),
) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "data": OmegaConf.load(data),
            "model": OmegaConf.load(model),
            "train": OmegaConf.load(train),
        }
    )
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(merged, DictConfig)
    OmegaConf.resolve(merged)
    return merged


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def env_info() -> dict[str, Any]:
    """Library versions, GPU and git commit, saved with every run (FR-T1, FR-T9)."""
    import albumentations
    import peft
    import timm

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timm": timm.__version__,
        "peft": peft.__version__,
        "albumentations": albumentations.__version__,
        "numpy": np.__version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


# ----------------------------------------------------------------------------- data


def _worker_init(_: int) -> None:
    info = get_worker_info()
    assert info is not None
    seed = info.seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
    transform = getattr(info.dataset, "transform", None)
    if isinstance(transform, TrainTransform):
        transform.reseed(seed)


def _every_nth_frame(df: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = [g.iloc[sample_indices(len(g), n)] for _, g in df.groupby("video_id", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else df


def make_data(cfg: DictConfig) -> tuple[FaceCropDataset, FaceCropDataset]:
    t = cfg.train
    manifest = Path(cfg.data.manifest_dir) / t.dataset / "manifest.parquet"
    kwargs = {"leave_out_method": t.leave_out_method, "compressions": list(t.compressions)}
    train_df = load_manifest(manifest, split="train", **kwargs)
    if t.sbi.only:
        train_df = train_df[train_df["label"] == 0].reset_index(drop=True)
    # The held-out method is dropped from validation too, so it never guides model selection.
    val_df = _every_nth_frame(
        load_manifest(manifest, split="val", **kwargs), t.val.frames_per_video
    )
    sbi_prob = t.sbi.prob if t.sbi.enabled else 0.0
    if t.sbi.only:
        sbi_prob = max(sbi_prob, 0.5)
    transform = TrainTransform(t.augment.preset, sbi_prob=sbi_prob, seed=t.seed)
    root = cfg.data.root
    return FaceCropDataset(train_df, root, transform), FaceCropDataset(val_df, root)


def epoch_indices(labels: np.ndarray, n: int, seed: int, epoch: int) -> list[int]:
    """Class-balanced sample order for one epoch; the same seed and epoch give the same order,
    which lets a run resume in the middle of an epoch."""
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = 1.0 / counts[labels]
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    w = torch.as_tensor(weights, dtype=torch.double)
    return torch.multinomial(w, n, replacement=True, generator=g).tolist()  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------- optimisation


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def cosine_with_warmup(
    opt: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


# ----------------------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, **state: Any) -> None:
    """Write atomically so a crash or sleep during saving never corrupts the last checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------- training


def train(
    cfg: DictConfig, resume: Path | None = None, pretrained_weights: bool = True
) -> dict[str, Any]:
    t = cfg.train
    ckpt: dict[str, Any] | None = None
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        cfg = OmegaConf.create(ckpt["config"])
        t = cfg.train
        run_dir = resume.parent
    else:
        name = t.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(t.out_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "config.yaml")
        (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))

    set_seed(t.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = t.precision == "bf16"

    train_ds, val_ds = make_data(cfg)
    labels = train_ds.df["label"].to_numpy().astype(int)
    n_samples = int(t.samples_per_epoch or len(train_ds))
    batches_per_epoch = math.ceil(n_samples / t.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / t.grad_accum)
    total_steps = steps_per_epoch * t.epochs
    warmup = int(t.scheduler.warmup_epochs * steps_per_epoch)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_cfg["grad_checkpointing"] = bool(t.grad_checkpointing)
    model = build_model(
        {str(k): v for k, v in model_cfg.items()},
        pretrained_weights=pretrained_weights and ckpt is None,
    )
    model.to(device)
    opt = make_optimizer(model, t.optimizer.lr, t.optimizer.weight_decay)
    sched = cosine_with_warmup(opt, warmup, total_steps)

    start_epoch, skip_batches, global_step, best_auc = 0, 0, 0, -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch, skip_batches = ckpt["epoch"], ckpt["batch_in_epoch"]
        global_step, best_auc = ckpt["global_step"], ckpt["best_val_auc"]
        log.info("resumed from %s at epoch %d batch %d", resume, start_epoch, skip_batches)

    trainable, total = count_parameters(model)
    log.info("params: %.2fM trainable of %.2fM; device %s", trainable / 1e6, total / 1e6, device)
    log.info(
        "train crops %d, val crops %d, %d optimizer steps per epoch",
        len(train_ds),
        len(val_ds),
        steps_per_epoch,
    )

    writer = SummaryWriter(str(run_dir / "tb"))
    loader_kw: dict[str, Any] = {
        "batch_size": t.batch_size,
        "num_workers": t.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if t.num_workers > 0:
        loader_kw["worker_init_fn"] = _worker_init
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    method_weight = float(cfg.model.method_head.weight)

    def checkpoint_state(epoch: int, batch_in_epoch: int, metrics: dict[str, float]) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "best_val_auc": best_auc,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": metrics,
        }

    last_save = time.monotonic()
    metrics: dict[str, float] = {}
    for epoch in range(start_epoch, t.epochs):
        model.train()
        order = epoch_indices(labels, n_samples, t.seed, epoch)
        consumed = skip_batches * t.batch_size if epoch == start_epoch else 0
        loader = DataLoader(train_ds, sampler=order[consumed:], **loader_kw)
        first_batch = consumed // t.batch_size
        opt.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        for i, batch in enumerate(loader, start=first_batch):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                out = model(x)
            loss = bce(out["logit"].float(), y)
            if "method_logits" in out:
                target = method_target(list(batch["method"])).to(device)
                loss = loss + method_weight * ce(out["method_logits"].float(), target)
            (loss / t.grad_accum).backward()
            running += loss.item()
            seen += 1

            if (i + 1) % t.grad_accum == 0 or i + 1 == batches_per_epoch:
                nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % t.log_every == 0:
                    writer.add_scalar("train/loss", running / seen, global_step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
                    running, seen = 0.0, 0
                if time.monotonic() - last_save > t.checkpoint.every_minutes * 60:
                    save_checkpoint(run_dir / "last.pt", **checkpoint_state(epoch, i + 1, metrics))
                    last_save = time.monotonic()

        frames = predict_frames(model, val_loader, device, bf16)
        metrics = summarize(frames, how=t.val.video_score)
        per_method = auc_per_method(frames, how=t.val.video_score)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch + 1)
        for k, v in per_method.items():
            writer.add_scalar(f"val_method_auc/{k}", v, epoch + 1)
        if torch.cuda.is_available():
            writer.add_scalar(
                "gpu/peak_mem_gb", torch.cuda.max_memory_allocated() / 2**30, epoch + 1
            )
        _log_samples(writer, val_ds, frames, epoch + 1)

        improved = metrics["video_auc"] > best_auc
        if improved:
            best_auc = metrics["video_auc"]
        state = checkpoint_state(epoch + 1, 0, metrics)
        save_checkpoint(run_dir / "last.pt", **state)
        if improved:
            save_checkpoint(run_dir / "best.pt", **state)
        last_save = time.monotonic()
        record = {"epoch": epoch + 1, "step": global_step, **metrics, "per_method": per_method}
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(
            "epoch %d: val video AUC %.4f (best %.4f)", epoch + 1, metrics["video_auc"], best_auc
        )

    writer.close()
    return {"run_dir": str(run_dir), "best_val_auc": best_auc, "last_metrics": metrics}


def _log_samples(
    writer: SummaryWriter, ds: FaceCropDataset, frames: pd.DataFrame, step: int
) -> None:
    """Eight validation crops with their scores (FR-T7)."""
    idx = np.linspace(0, len(ds) - 1, num=min(8, len(ds))).astype(int)
    if len(idx) == 0:
        return
    images = torch.stack([ds[int(i)]["image"] for i in idx])
    writer.add_images("val/samples", images, step)
    rows = frames.iloc[idx]
    lines = [
        f"{r.video_id} label={r.label} p_fake={r.prob:.3f}" for r in rows.itertuples(index=False)
    ]
    writer.add_text("val/sample_scores", "  \n".join(lines), step)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--model", default="configs/model.yaml")
    p.add_argument("--train", default="configs/train.yaml")
    p.add_argument("--resume", type=Path, default=None, help="path to a last.pt or best.pt")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=5")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.data, args.model, args.train, args.overrides)
    result = train(cfg, resume=args.resume)
    log.info("done: %s", result)


if __name__ == "__main__":
    main()
