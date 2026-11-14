"""Teach a trained model with examples marked in the demo.

When the demo gets something wrong, "Teach the model" saves that file (and only that file) under
data/feedback/real/ or data/feedback/fake/user/ on this computer. This tool then:

1. finds and crops the faces in every saved example, with the same face pipeline as training;
2. fine-tunes a copy of the base model on those examples, repeated, mixed with a sample of its
   original training data so it does not forget what it already knew;
3. checks the copy on the base model's validation set and keeps it only if validation AUC drops
   by no more than --max-drop; and
4. recalibrates it and writes runs/<base run>_taught/best.pt, which the demo then picks up.

A model taught on a few examples will learn those examples. That is useful for catching the
same video again, but it is not evidence that it now catches deepfakes in general: only a
held-out benchmark can show that.

    python -m deepfake_detector.teach                      # base: runs/open_v1/best.pt
    python -m deepfake_detector.teach --base runs/<run>/best.pt --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

FEEDBACK_ROOT = Path("data/feedback")
LABEL_DIRS = {"real": Path("real"), "fake": Path("fake") / "user"}
DEFAULT_BASE = Path("runs/open_v1/best.pt")


# ----------------------------------------------------------------------------- saving examples


def save_example(data: bytes, suffix: str, label: str, root: Path = FEEDBACK_ROOT) -> Path:
    """Store one marked example under a content hash, so the same file is kept once.

    Marking a file again with the other label moves it, so the latest answer wins.
    """
    if label not in LABEL_DIRS:
        raise ValueError(f"label must be one of {sorted(LABEL_DIRS)}")
    suffix = suffix.lower()
    if not suffix.startswith(".") or not suffix[1:].isalnum():
        raise ValueError("bad file suffix")
    name = hashlib.sha256(data).hexdigest()[:20] + suffix
    for other, sub in LABEL_DIRS.items():
        if other != label:
            (root / sub / name).unlink(missing_ok=True)
    dest = root / LABEL_DIRS[label] / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return dest


def example_counts(root: Path = FEEDBACK_ROOT) -> dict[str, int]:
    return {
        label: sum(1 for p in (root / sub).glob("*") if p.is_file()) if (root / sub).exists() else 0
        for label, sub in LABEL_DIRS.items()
    }


# ----------------------------------------------------------------------------- data


def prepare(root: Path, data_root: Path, frames: int = 8, workers: int = 1) -> pd.DataFrame:
    """Crop faces from every saved example and return their manifest rows."""
    from deepfake_detector.data.indexers import index_folders
    from deepfake_detector.data.preprocess import finalize, run

    records = index_folders(root, "feedback", split="train")
    if not records:
        raise SystemExit(f"No examples in {root}. Mark some in the demo first.")
    run(records, data_root, frames=frames, workers=workers)
    crops, _ = finalize("feedback", data_root)
    # An example marked again with the other label moved folders; drop its stale crops.
    crops = crops[crops["video_id"].isin({r.video_id for r in records})]
    if crops.empty:
        raise SystemExit("No face was found in any marked example, so there is nothing to learn.")
    return crops.reset_index(drop=True)


def mix(
    examples: pd.DataFrame, replay: pd.DataFrame, repeats: int, n_replay: int, seed: int
) -> pd.DataFrame:
    """The examples repeated `repeats` times plus a class-balanced sample of the old data."""
    parts = [examples] * repeats
    for _, g in replay.groupby("label"):
        parts.append(g.sample(min(len(g), n_replay // 2), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)


# ----------------------------------------------------------------------------- teaching


def teach(
    base: Path = DEFAULT_BASE,
    feedback_root: Path = FEEDBACK_ROOT,
    epochs: int = 3,
    repeats: int = 20,
    n_replay: int = 3000,
    lr: float = 5e-5,
    max_drop: float = 0.01,
    batch_size: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    from deepfake_detector.calibrate import Calibration, calibrate_frames
    from deepfake_detector.data.augment import TrainTransform
    from deepfake_detector.data.datasets import FaceCropDataset, load_manifests
    from deepfake_detector.inference import load_detector, predict_frames
    from deepfake_detector.metrics import summarize
    from deepfake_detector.train import _every_nth_frame, save_checkpoint

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda"
    model, cfg = load_detector(base, device)
    t = cfg["train"]
    data_root = Path(cfg["data"]["root"])
    man_dir, names = cfg["data"]["manifest_dir"], t["dataset"]
    filters = {"leave_out_method": t["leave_out_method"], "compressions": t["compressions"]}

    examples = prepare(feedback_root, data_root)
    val_df = _every_nth_frame(
        load_manifests(man_dir, names, split="val", **filters), t["val"]["frames_per_video"]
    )

    def score(df: pd.DataFrame) -> pd.DataFrame:
        loader = DataLoader(FaceCropDataset(df, data_root), batch_size=64, shuffle=False)
        return predict_frames(model, loader, device, bf16)

    base_cal_path = base.parent / "calibration.json"
    base_cal = Calibration.load(base_cal_path) if base_cal_path.exists() else Calibration()
    how = t["val"]["video_score"]
    val_before = summarize(score(val_df), how=how)
    ex_before = score(examples)

    sbi_prob = float(t["sbi"]["prob"]) if t["sbi"]["enabled"] else 0.0
    train_df = mix(
        examples, load_manifests(man_dir, names, split="train", **filters), repeats, n_replay, seed
    )
    ds = FaceCropDataset(
        train_df, data_root, TrainTransform(t["augment"]["preset"], sbi_prob=sbi_prob, seed=seed)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(t["optimizer"]["weight_decay"]))
    bce = nn.BCEWithLogitsLoss()
    log.info(
        "teaching on %d example crops (x%d) plus %d old crops, %d epochs",
        len(examples), repeats, len(train_df) - len(examples) * repeats, epochs,
    )  # fmt: skip
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                loss = bce(model(x)["logit"].float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
            opt.step()
            total, n = total + loss.item(), n + 1
        log.info("epoch %d: loss %.4f", epoch + 1, total / max(1, n))

    val_frames = score(val_df)
    val_after = summarize(val_frames, how=how)
    cal = calibrate_frames(val_frames, how)
    cal.fitted_on = f"{t['dataset']} val after teaching"
    ex_after = score(examples)

    def per_example(frames: pd.DataFrame, c: Calibration) -> dict[str, float]:
        f = frames.assign(prob=c.apply(frames["logit"].to_numpy()))
        return {str(k): float(v) for k, v in f.groupby("video_id")["prob"].mean().items()}

    def mean_logit(frames: pd.DataFrame) -> dict[str, float]:
        return {str(k): float(v) for k, v in frames.groupby("video_id")["logit"].mean().items()}

    before_p, after_p = per_example(ex_before, base_cal), per_example(ex_after, cal)
    before_l, after_l = mean_logit(ex_before), mean_logit(ex_after)
    labels = examples.groupby("video_id")["label"].first()
    drop = val_before["video_auc"] - val_after["video_auc"]
    report: dict[str, Any] = {
        "base": str(base),
        "examples": example_counts(feedback_root),
        "val_auc_before": val_before["video_auc"],
        "val_auc_after": val_after["video_auc"],
        "kept": bool(drop <= max_drop),
        "max_drop": max_drop,
        "per_example": [
            {
                "id": k,
                "label": int(labels[k]),
                "before": before_p[k],
                "after": after_p.get(k),
                "logit_before": before_l[k],
                "logit_after": after_l.get(k),
            }
            for k in before_p
        ],
    }
    out_dir = base.parent.parent / f"{base.parent.name}_taught"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teach.json").write_text(json.dumps(report, indent=2))
    if report["kept"]:
        cfg = {**cfg, "train": {**t, "taught": {"base": str(base), **report["examples"]}}}
        save_checkpoint(
            out_dir / "best.pt",
            model=model.state_dict(),
            config=cfg,
            metrics=val_after,
            epoch=epochs,
            batch_in_epoch=0,
            global_step=0,
            best_val_auc=val_after["video_auc"],
        )
        cal.save(out_dir / "calibration.json")
        log.info("kept: val AUC %.4f -> %.4f", val_before["video_auc"], val_after["video_auc"])
    else:
        (out_dir / "best.pt").unlink(missing_ok=True)
        log.warning(
            "not kept: val AUC fell %.4f -> %.4f (limit %.3f); the base model stays in use",
            val_before["video_auc"], val_after["video_auc"], max_drop,
        )  # fmt: skip
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--feedback", type=Path, default=FEEDBACK_ROOT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20, help="times each example is repeated")
    p.add_argument("--replay", type=int, default=3000, help="old training crops mixed in")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-drop", type=float, default=0.01, help="largest val AUC loss allowed")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = teach(
        args.base, args.feedback, args.epochs, args.repeats, args.replay, args.lr, args.max_drop
    )
    rows = [
        f"  {'fake' if e['label'] else 'real'}  {e['before']:.1%} -> "
        + (f"{e['after']:.1%}" if e["after"] is not None else "no face")
        for e in report["per_example"]
    ]
    print("\n".join([f"kept: {report['kept']}", "examples (before -> after):", *rows]))


if __name__ == "__main__":
    main()
