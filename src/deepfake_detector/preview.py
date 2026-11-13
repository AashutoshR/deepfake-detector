"""Build a preview model that runs before any deepfake dataset arrives.

Self-blended images need only real faces, so a detector can be trained on public real photos
alone. This tool downloads part of FFHQ (real faces, CC BY-NC-SA 4.0) from the Hugging Face hub,
preprocesses it with the same face pipeline, trains the ViT with LoRA on self-blended fakes only,
calibrates on fixed self-blended validation fakes, and scores the held-out test photos.

The preview model is for trying the demo. It has never seen a real deepfake and has not been
tested on any deepfake benchmark, so it is never released as the project model.

    python -m deepfake_detector.preview all --shards 8          # every step below, in order
    python -m deepfake_detector.preview download --shards 8
    python -m deepfake_detector.preview preprocess
    python -m deepfake_detector.preview train --epochs 8
    python -m deepfake_detector.preview finish                  # calibrate, test, write PREVIEW.md
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

FFHQ_REPO = "gaunernst/ffhq-1024-wds"  # 70 shards of 1,000 aligned 1024 px faces each
RUN_NAME = "preview_ffhq_sbi"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PREVIEW_NOTE = (
    "Preview model: trained only on self-blended fakes made from FFHQ real photos. It has not "
    "seen a real deepfake and has not been tested on any deepfake benchmark."
)


# ----------------------------------------------------------------------------- download


def shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def unpack_shard(tar_path: Path, out_dir: Path, max_side: int = 512) -> int:
    """Write every image in one shard as out_dir/<id>.jpg, shrunk to max_side.

    Members are read in memory and only image files are kept, so nothing from the archive is
    written under a name it chose.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = Path(m.name)
            if not m.isfile() or name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = "".join(c for c in name.stem if c.isalnum()) or f"img{n}"
            dest = out_dir / f"{stem}.jpg"
            if dest.exists():
                n += 1
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imwrite(str(dest), shrink(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
    return n


def download(shards: int, raw: Path, max_side: int = 512, keep: bool = False) -> int:
    """Fetch the first `shards` FFHQ shards, one at a time, deleting each after unpacking."""
    from huggingface_hub import hf_hub_download

    out = raw / "ffhq" / "real"
    tmp = raw.parent / "downloads" / "ffhq"
    total = 0
    for i in range(shards):
        name = f"{i * 1000:05d}.tar"
        done = out / f".{name}.done"
        if done.exists():
            total += int(done.read_text())
            continue
        log.info("shard %d/%d: %s", i + 1, shards, name)
        path = Path(hf_hub_download(FFHQ_REPO, name, repo_type="dataset", local_dir=str(tmp)))
        n = unpack_shard(path, out, max_side)
        done.write_text(str(n))
        total += n
        if not keep:
            path.unlink(missing_ok=True)
    log.info("%d FFHQ photos in %s", total, out)
    return total


# ----------------------------------------------------------------------------- steps


def preprocess(raw: Path, data_root: Path, workers: int = 2) -> None:
    from deepfake_detector.data.indexers import index_ffhq
    from deepfake_detector.data.preprocess import finalize, run

    records = index_ffhq(raw / "ffhq")
    counts = run(records, data_root, frames=1, workers=workers)
    log.info("preprocess counts: %s", counts)
    finalize("ffhq", data_root)


def train_overrides(epochs: int, run_name: str = RUN_NAME) -> list[str]:
    return [
        "train.dataset=ffhq",
        "train.compressions=[na]",
        "train.sbi.only=true",
        "train.sbi.prob=0.5",
        "train.val.sbi_fakes=true",
        "train.val.frames_per_video=1",
        f"train.epochs={epochs}",
        f"train.run_name={run_name}",
    ]


def train(epochs: int, run_name: str = RUN_NAME) -> dict[str, Any]:
    from deepfake_detector.train import load_config
    from deepfake_detector.train import train as run_training

    run_dir = Path("runs") / run_name
    last = run_dir / "last.pt"
    if last.exists():
        log.info("resuming %s", last)
        return run_training(load_config(), resume=last)
    return run_training(load_config(overrides=train_overrides(epochs, run_name)))


def finish(run_name: str = RUN_NAME) -> dict[str, Any]:
    """Calibrate on validation, score the test photos, and write PREVIEW.md next to the run."""
    from deepfake_detector.calibrate import calibrate_frames
    from deepfake_detector.evaluate import score_split
    from deepfake_detector.metrics import summarize

    ckpt = Path("runs") / run_name / "best.pt"
    val, cfg = score_split(ckpt, split="val")
    cal = calibrate_frames(val, cfg["train"]["val"]["video_score"])
    cal.fitted_on = "ffhq val (self-blended fakes)"
    cal.save(ckpt.parent / "calibration.json")

    test, _ = score_split(ckpt, split="test")
    test = test.assign(prob=cal.apply(test["logit"].to_numpy()))
    metrics = summarize(test, how="mean")
    report = {
        "checkpoint": str(ckpt),
        "temperature": cal.temperature,
        "bands": [cal.likely_real_below, cal.likely_fake_above],
        "test_photos": int((test["label"] == 0).sum()),
        "test_metrics_self_blended": metrics,
        "note": PREVIEW_NOTE,
    }
    (ckpt.parent / "preview.json").write_text(json.dumps(report, indent=2))
    (ckpt.parent / "PREVIEW.md").write_text(preview_markdown(report))
    log.info("preview test (self-blended fakes): %s", metrics)
    return report


def preview_markdown(r: dict[str, Any]) -> str:
    m = r["test_metrics_self_blended"]
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in m.items() if isinstance(v, float))
    lo, hi = r["bands"]
    return (
        "# Preview model\n\n"
        f"{r['note']}\n\n"
        f"Checkpoint: `{r['checkpoint']}`; temperature {r['temperature']:.3f}; bands: likely "
        f"real below {lo:.3f}, likely manipulated above {hi:.3f}.\n\n"
        f"Held-out test: {r['test_photos']} FFHQ photos plus a fixed self-blended fake of each. "
        "These numbers only show the model learned to spot blending; they say nothing about "
        "real deepfakes.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"{rows}\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["all", "download", "preprocess", "train", "finish"])
    p.add_argument("--shards", type=int, default=8, help="FFHQ shards of 1,000 photos (~1 GB)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.step in ("all", "download"):
        download(args.shards, args.raw)
    if args.step in ("all", "preprocess"):
        preprocess(args.raw, args.data, args.workers)
    if args.step in ("all", "train"):
        train(args.epochs)
    if args.step in ("all", "finish"):
        finish()


if __name__ == "__main__":
    main()
