"""Build a preview model that runs before any deepfake dataset arrives.

Self-blended images need only real faces, so a detector can be trained on public real photos
alone. This tool downloads part of FFHQ (real faces, CC BY-NC-SA 4.0) from the Hugging Face hub,
preprocesses it with the same face pipeline, trains the ViT with LoRA on self-blended fakes only,
calibrates on fixed self-blended validation fakes, and scores the held-out test photos.

A second preview (--source dff) adds real fakes that need no access request: DeepFakeFace
(OpenRAIL), 30,000 IMDB-WIKI celebrity photos with an InsightFace face swap, a Stable Diffusion
inpainting and a text-to-image fake for each. It trains on the swaps and inpaintings plus
self-blended fakes and keeps text-to-image fakes unseen as a cross-generator test.

Preview models are for trying the demo while the FaceForensics++ data is pending. Neither is
tested on a video deepfake benchmark, so neither is ever released as the project model.

    python -m deepfake_detector.preview all --source dff --every 6 --epochs 8
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
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FFHQ_REPO = "gaunernst/ffhq-1024-wds"  # 70 shards of 1,000 aligned 1024 px faces each
RUN_NAME = "preview_ffhq_sbi"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
PREVIEW_NOTE = (
    "Preview model: trained only on self-blended fakes made from FFHQ real photos. It has not "
    "seen a real deepfake and has not been tested on any deepfake benchmark."
)
DFF_REPO = "OpenRL/DeepFakeFace"  # OpenRAIL; 30,000 images per zip
DFF_ZIPS = ("wiki.zip", "insight.zip", "inpainting.zip", "text2img.zip")
DFF_RUN_NAME = "preview_dff"
DFF_HELD_OUT = "text2img"
DFF_NOTE = (
    "Preview model: trained on DeepFakeFace photos (InsightFace face swaps and Stable Diffusion "
    "inpainting of celebrity photos) plus self-blended fakes. It has not been trained or tested "
    "on video deepfake benchmarks such as FaceForensics++ or Celeb-DF, so polished or heavily "
    "re-encoded deepfakes can still score as real."
)
NOTES = {"ffhq": PREVIEW_NOTE, "dff": DFF_NOTE}


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


# ----------------------------------------------------------------------------- DeepFakeFace


def unpack_dff_zip(zip_path: Path, out_root: Path, every: int = 1) -> int:
    """Write the images of one DeepFakeFace zip under out_root/<folder>/<xx>/<name>.jpg.

    Members are read in memory and every path part is checked and cleaned, so nothing is written
    under a name the archive chose. With every=k only persons whose id // 20 is a multiple of k
    are kept, which shrinks the set without changing the split ratios (splits use id % 20).
    """
    from deepfake_detector.data.indexers import DFF_FOLDERS

    def clean(s: str) -> str:
        return "".join(c for c in s if c.isalnum() or c in "_-")

    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if info.is_dir() or len(parts) != 3 or parts[0] not in DFF_FOLDERS:
                continue
            name = PurePosixPath(parts[2])
            if name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            person = name.stem.split("_")[0]
            if every > 1 and person.isdigit() and (int(person) // 20) % every:
                continue
            sub, stem = clean(parts[1]), clean(name.stem)
            if not sub or not stem:
                continue
            dest = out_root / parts[0] / sub / f"{stem}.jpg"
            if dest.exists():
                n += 1
                continue
            data = zf.read(info)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name.suffix.lower() in (".jpg", ".jpeg"):
                dest.write_bytes(data)  # keep the original bytes: re-encoding hides artifacts
            else:
                cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
    return n


def download_dff(raw: Path, every: int = 6, keep: bool = False) -> int:
    """Fetch the four DeepFakeFace zips (about 5 GB) one at a time and unpack them."""
    from huggingface_hub import hf_hub_download

    out = raw / "dff"
    tmp = raw.parent / "downloads" / "dff"
    total = 0
    for name in DFF_ZIPS:
        done = out / f".{name}.every{every}.done"
        # an earlier unpack with a k that divides `every` already holds every person needed
        have = [out / f".{name}.every{k}.done" for k in range(1, every + 1) if every % k == 0]
        if any(m.exists() for m in have):
            total += int(next(m for m in have if m.exists()).read_text())
            continue
        log.info("DeepFakeFace %s", name)
        path = Path(hf_hub_download(DFF_REPO, name, repo_type="dataset", local_dir=str(tmp)))
        n = unpack_dff_zip(path, out, every)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(str(n))
        total += n
        if not keep:
            path.unlink(missing_ok=True)
    log.info("%d DeepFakeFace images in %s", total, out)
    return total


# ----------------------------------------------------------------------------- steps


def dff_keep(rec: Any, every: int) -> bool:
    """Persons kept at subsampling `every`; held-out fakes are only needed in the test split."""
    if rec.method == DFF_HELD_OUT and rec.split != "test":
        return False
    person = rec.persons[0].split(":", 1)[1]
    return every <= 1 or not person.isdigit() or (int(person) // 20) % every == 0


def preprocess(
    raw: Path, data_root: Path, workers: int = 2, source: str = "ffhq", every: int = 1
) -> None:
    from deepfake_detector.data.indexers import INDEXERS
    from deepfake_detector.data.preprocess import finalize, run

    records = INDEXERS[source](raw / source)
    if source == "dff":
        records = [r for r in records if dff_keep(r, every)]
    counts = run(records, data_root, frames=1, workers=workers)
    log.info("preprocess counts: %s", counts)
    finalize(source, data_root)


def train_overrides(epochs: int, run_name: str = RUN_NAME, source: str = "ffhq") -> list[str]:
    if source == "dff":
        return [
            "train.dataset=dff",
            "train.compressions=[na]",
            "train.sbi.only=false",
            "train.sbi.prob=0.3",
            f"train.leave_out_method={DFF_HELD_OUT}",
            "train.val.frames_per_video=1",
            f"train.epochs={epochs}",
            f"train.run_name={run_name}",
        ]
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


def train(epochs: int, run_name: str = RUN_NAME, source: str = "ffhq") -> dict[str, Any]:
    from deepfake_detector.train import load_config
    from deepfake_detector.train import train as run_training

    run_dir = Path("runs") / run_name
    last = run_dir / "last.pt"
    if last.exists():
        log.info("resuming %s", last)
        return run_training(load_config(), resume=last)
    return run_training(load_config(overrides=train_overrides(epochs, run_name, source)))


def finish(run_name: str = RUN_NAME, source: str = "ffhq") -> dict[str, Any]:
    """Calibrate on validation, score the test split, and write PREVIEW.md next to the run."""
    from deepfake_detector.calibrate import calibrate_frames
    from deepfake_detector.evaluate import score_split
    from deepfake_detector.metrics import summarize

    ckpt = Path("runs") / run_name / "best.pt"
    val, cfg = score_split(ckpt, split="val")
    cal = calibrate_frames(val, cfg["train"]["val"]["video_score"])
    cal.fitted_on = "dff val" if source == "dff" else "ffhq val (self-blended fakes)"
    cal.save(ckpt.parent / "calibration.json")

    test, _ = score_split(ckpt, split="test", keep_left_out=True)
    test = test.assign(prob=cal.apply(test["logit"].to_numpy()))
    real = test[test["label"] == 0]
    per_method = {
        str(m): summarize(pd.concat([real, g]), how="mean")
        for m, g in test[test["label"] == 1].groupby("method")
    }
    report = {
        "source": source,
        "checkpoint": str(ckpt),
        "temperature": cal.temperature,
        "bands": [cal.likely_real_below, cal.likely_fake_above],
        "test_photos": int(real["video_id"].nunique()),
        "held_out_method": cfg["train"].get("leave_out_method"),
        "test_metrics": per_method,
        "note": NOTES[source],
    }
    (ckpt.parent / "preview.json").write_text(json.dumps(report, indent=2))
    (ckpt.parent / "PREVIEW.md").write_text(preview_markdown(report))
    log.info("preview test: %s", per_method)
    return report


def preview_markdown(r: dict[str, Any]) -> str:
    lo, hi = r["bands"]
    held = r.get("held_out_method")
    out = [
        "# Preview model\n\n",
        f"{r['note']}\n\n",
        f"Checkpoint: `{r['checkpoint']}`; temperature {r['temperature']:.3f}; bands: likely "
        f"real below {lo:.3f}, likely manipulated above {hi:.3f}.\n\n",
        f"Held-out test: {r['test_photos']} real photos of people the model never trained on, "
        "each against every kind of fake below.",
    ]
    if held:
        out.append(f" `{held}` fakes were never used in training or validation.")
    out.append("\n\n| Fakes | AUC | EER | Accuracy | ECE | Videos |\n|---|---|---|---|---|---|\n")
    for method, m in r["test_metrics"].items():
        name = f"{method} (unseen)" if method == held else method
        out.append(
            f"| {name} | {m['video_auc']:.4f} | {m['video_eer']:.4f} | {m['video_acc']:.4f} "
            f"| {m['video_ece']:.4f} | {int(m['n_videos'])} |\n"
        )
    return "".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["all", "download", "preprocess", "train", "finish"])
    p.add_argument("--source", choices=["ffhq", "dff"], default="ffhq")
    p.add_argument("--shards", type=int, default=8, help="FFHQ shards of 1,000 photos (~1 GB)")
    p.add_argument("--every", type=int, default=6, help="DeepFakeFace: keep 1 in k persons")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_name = RUN_NAME if args.source == "ffhq" else DFF_RUN_NAME
    if args.step in ("all", "download"):
        if args.source == "dff":
            download_dff(args.raw, args.every)
        else:
            download(args.shards, args.raw)
    if args.step in ("all", "preprocess"):
        preprocess(args.raw, args.data, args.workers, args.source, args.every)
    if args.step in ("all", "train"):
        train(args.epochs, run_name, args.source)
    if args.step in ("all", "finish"):
        finish(run_name, args.source)


if __name__ == "__main__":
    main()
