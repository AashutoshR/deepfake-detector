"""Train the detector on every openly licensed deepfake dataset, while FF++ access is pending.

Training data (all split by person or by source video, so no face is in two splits):
    DeepFakeFace   IMDB-WIKI photos with InsightFace swaps, SD inpaintings, SD text-to-image
    OpenForensics  in-the-wild photos, each face labelled real or GAN face-swapped
    WildDeepfake   face sequences from real and fake videos collected on the internet
plus self-blended fakes made from the real faces during training.

Steps (each can be rerun; finished work is skipped):

    python -m deepfake_detector.open_model all
    python -m deepfake_detector.open_model fetch        # WildDeepfake + OpenForensics import
    python -m deepfake_detector.open_model preprocess   # face crops for all three
    python -m deepfake_detector.open_model train --epochs 10
    python -m deepfake_detector.open_model finish       # calibrate, test each set, MODEL.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DATASETS = ("dff", "openforensics", "wilddeepfake")
RUN_NAME = "open_v1"
DFF_EVERY = 3
OPEN_NOTE = (
    "Trained on openly licensed data: DeepFakeFace, OpenForensics and WildDeepfake, plus "
    "self-blended fakes. Not yet trained or tested on FaceForensics++ or Celeb-DF. A low score "
    "means this model found nothing, not that the media is real."
)


def fetch(raw: Path, downloads: Path, frames: int = 6, of_every: int = 3) -> None:
    from deepfake_detector.data.open_sets import fetch_wilddeepfake, import_openforensics

    import_openforensics(downloads, raw, {"train": of_every, "val": of_every, "test": of_every})
    fetch_wilddeepfake(raw, frames)


def preprocess(raw: Path, data_root: Path, workers: int = 2, frames: int = 6) -> None:
    from deepfake_detector.data.indexers import INDEXERS
    from deepfake_detector.data.preprocess import finalize, run

    for name in DATASETS:
        records = INDEXERS[name](raw / name)
        if name == "dff":  # every DeepFakeFace kind, text-to-image too, is trained on here
            records = [r for r in records if _dff_person_kept(r)]
        log.info("%s: %d items", name, len(records))
        run(records, data_root, frames=frames if name == "wilddeepfake" else 1, workers=workers)
        finalize(name, data_root)


def _dff_person_kept(rec: Any) -> bool:
    person = rec.persons[0].split(":", 1)[1]
    return not person.isdigit() or (int(person) // 20) % DFF_EVERY == 0


def train_overrides(epochs: int, run_name: str = RUN_NAME) -> list[str]:
    return [
        f"train.dataset=[{','.join(DATASETS)}]",
        "train.compressions=[na]",
        "train.sbi.enabled=true",
        "train.sbi.only=false",
        "train.sbi.prob=0.3",
        "train.leave_out_method=null",
        "train.val.frames_per_video=2",
        f"train.epochs={epochs}",
        f"train.run_name={run_name}",
    ]


def train(epochs: int, run_name: str = RUN_NAME) -> dict[str, Any]:
    from deepfake_detector.train import load_config
    from deepfake_detector.train import train as run_training

    last = Path("runs") / run_name / "last.pt"
    if last.exists():
        log.info("resuming %s", last)
        return run_training(load_config(), resume=last)
    return run_training(load_config(overrides=train_overrides(epochs, run_name)))


def finish(run_name: str = RUN_NAME) -> dict[str, Any]:
    """Calibrate on the joint validation split, then test every dataset and fake kind."""
    from deepfake_detector.calibrate import calibrate_frames
    from deepfake_detector.evaluate import score_split
    from deepfake_detector.metrics import summarize

    ckpt = Path("runs") / run_name / "best.pt"
    val, cfg = score_split(ckpt, split="val")
    how = cfg["train"]["val"]["video_score"]
    cal = calibrate_frames(val, how)
    cal.fitted_on = "joint val: " + ", ".join(DATASETS)
    cal.save(ckpt.parent / "calibration.json")

    test, _ = score_split(ckpt, split="test")
    test = test.assign(prob=cal.apply(test["logit"].to_numpy()))
    test["dataset"] = test["video_id"].map(_dataset_of(cfg))
    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "temperature": cal.temperature,
        "bands": [cal.likely_real_below, cal.likely_fake_above],
        "val": summarize(val.assign(prob=cal.apply(val["logit"].to_numpy())), how=how),
        "test": {},
        "note": OPEN_NOTE,
    }
    for name, g in test.groupby("dataset"):
        real = g[g["label"] == 0]
        entry = {"all": summarize(g, how=how)}
        for method, f in g[g["label"] == 1].groupby("method"):
            entry[str(method)] = summarize(pd.concat([real, f]), how=how)
        report["test"][str(name)] = entry
    (ckpt.parent / "results.json").write_text(json.dumps(report, indent=2))
    (ckpt.parent / "MODEL.md").write_text(model_markdown(report))
    log.info("test: %s", {k: v["all"]["video_auc"] for k, v in report["test"].items()})
    return report


def _dataset_of(cfg: dict[str, Any]) -> dict[str, str]:
    """video_id -> dataset, read back from the manifests (video ids are unique per dataset)."""
    out: dict[str, str] = {}
    base = Path(cfg["data"]["manifest_dir"])
    for name in DATASETS:
        df = pd.read_parquet(base / name / "manifest.parquet", columns=["video_id"])
        out.update(dict.fromkeys(df["video_id"].unique(), name))
    return out


def model_markdown(r: dict[str, Any]) -> str:
    lo, hi = r["bands"]
    lines = [
        "# Open-data model\n\n",
        f"{r['note']}\n\n",
        f"Checkpoint `{r['checkpoint']}`; temperature {r['temperature']:.3f}; no manipulation "
        f"detected below {lo:.3f}, likely manipulated above {hi:.3f}.\n\n",
        "Held-out test splits (people or source videos never used in training). For video "
        "sets, one score per face sequence (mean of its frames).\n\n",
        "| Test set | Fakes | AUC | EER | Accuracy | ECE | Items |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for name, entry in r["test"].items():
        for kind, m in entry.items():
            lines.append(
                f"| {name} | {kind} | {m['video_auc']:.4f} | {m['video_eer']:.4f} | "
                f"{m['video_acc']:.4f} | {m['video_ece']:.4f} | {int(m['n_videos'])} |\n"
            )
    return "".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["all", "fetch", "preprocess", "train", "finish"])
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--downloads", type=Path, default=Path("data/downloads/openforensics"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.step in ("all", "fetch"):
        fetch(args.raw, args.downloads)
    if args.step in ("all", "preprocess"):
        preprocess(args.raw, args.data, args.workers)
    if args.step in ("all", "train"):
        train(args.epochs)
    if args.step in ("all", "finish"):
        finish()


if __name__ == "__main__":
    main()
