"""Train the detector on every openly licensed deepfake dataset, while FF++ access is pending.

Training data (all split by person or by source video, so no face is in two splits):
    DeepFakeFace   IMDB-WIKI photos with InsightFace swaps, SD inpaintings, SD text-to-image
    OpenForensics  in-the-wild photos, each face labelled real or GAN face-swapped
    WildDeepfake   face sequences from real and fake videos collected on the internet
    FFHQ           (v2) real face photos, to cut false alarms on ordinary photos
plus self-blended fakes made from the real faces during training.

Steps (each can be rerun; finished work is skipped):

    python -m deepfake_detector.open_model all --recipe v2
    python -m deepfake_detector.open_model fetch --recipe v2       # WildDeepfake, OpenForensics
    python -m deepfake_detector.open_model preprocess --recipe v2  # face crops
    python -m deepfake_detector.open_model train --recipe v2
    python -m deepfake_detector.open_model finish --recipe v2 --compare open_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

DATASETS = ("dff", "openforensics", "wilddeepfake")
RUN_NAME = "open_v1"
OPEN_NOTE = (
    "Trained on openly licensed data: DeepFakeFace, OpenForensics and WildDeepfake, plus "
    "self-blended fakes. Not yet trained or tested on FaceForensics++ or Celeb-DF. A low score "
    "means this model found nothing, not that the media is real."
)


@dataclass(frozen=True)
class Recipe:
    """Everything that differs between open-data model versions."""

    run_name: str
    datasets: tuple[str, ...]  # manifest names; wilddeepfake_f<k> keeps k frames per sequence
    dff_every: int  # keep 1 in k DeepFakeFace persons
    of_every: dict[str, int] = field(default_factory=dict)  # OpenForensics: 1 in k photos
    wdf_frames: int = 6
    lora_targets: tuple[str, ...] = ("qkv",)
    epochs: int = 10
    samples_per_epoch: int | None = None


RECIPES = {
    "v1": Recipe("open_v1", DATASETS, 3, {"train": 3, "val": 3, "test": 3}),
    # v2: all of DeepFakeFace and OpenForensics train, twice the WildDeepfake frames, FFHQ real
    # photos against false alarms, and LoRA on the attention output and MLP layers as well.
    "v2": Recipe(
        "open_v2",
        ("dff", "openforensics", "wilddeepfake_f12", "ffhq"),
        1,
        {"train": 1, "val": 1, "test": 3},
        wdf_frames=12,
        lora_targets=("qkv", "attn.proj", "fc1", "fc2"),
        epochs=8,
        samples_per_epoch=200_000,
    ),
}


def fetch(raw: Path, downloads: Path, r: Recipe) -> None:
    from deepfake_detector.data.open_sets import fetch_wilddeepfake, import_openforensics

    import_openforensics(downloads, raw, r.of_every)
    fetch_wilddeepfake(raw, r.wdf_frames)


def preprocess(raw: Path, data_root: Path, r: Recipe, workers: int = 2) -> None:
    from deepfake_detector.data.indexers import INDEXERS
    from deepfake_detector.data.preprocess import finalize, run

    for name in r.datasets:
        source = "wilddeepfake" if name.startswith("wilddeepfake") else name
        records = INDEXERS[source](raw / source)
        if source == "dff":  # every DeepFakeFace kind, text-to-image too, is trained on here
            records = [x for x in records if person_kept(x, r.dff_every)]
        if name != source:  # its own crops and manifest, so the extra frames really get cropped
            records = [replace(x, dataset=name) for x in records]
        frames = r.wdf_frames if source == "wilddeepfake" else 1
        log.info("%s: %d items", name, len(records))
        run(records, data_root, frames=frames, workers=workers)
        finalize(name, data_root)


def person_kept(rec: Any, every: int) -> bool:
    person = rec.persons[0].split(":", 1)[1]
    return every <= 1 or not person.isdigit() or (int(person) // 20) % every == 0


def train_overrides(epochs: int, run_name: str = RUN_NAME, r: Recipe | None = None) -> list[str]:
    r = r or RECIPES["v1"]
    out = [
        f"train.dataset=[{','.join(r.datasets)}]",
        "train.compressions=[na]",
        "train.sbi.enabled=true",
        "train.sbi.only=false",
        "train.sbi.prob=0.3",
        "train.leave_out_method=null",
        "train.val.frames_per_video=2",
        f"train.epochs={epochs}",
        f"train.run_name={run_name}",
        f"model.finetune.lora.target_modules=[{','.join(r.lora_targets)}]",
    ]
    if r.samples_per_epoch:
        out.append(f"train.samples_per_epoch={r.samples_per_epoch}")
    return out


def train(r: Recipe, epochs: int | None = None) -> dict[str, Any]:
    from deepfake_detector.train import load_config
    from deepfake_detector.train import train as run_training

    last = Path("runs") / r.run_name / "last.pt"
    if last.exists():
        log.info("resuming %s", last)
        return run_training(load_config(), resume=last)
    overrides = train_overrides(epochs or r.epochs, r.run_name, r)
    return run_training(load_config(overrides=overrides))


def decision_metrics(videos: pd.DataFrame, lo: float, hi: float) -> dict[str, float]:
    """How the three bands behave: flagged above hi, cleared below lo, uncertain between.

    precision: share of flagged items that are fake; recall: share of fakes flagged;
    false_alarm_rate: share of real items flagged; missed_rate: share of fakes cleared.
    """
    y, s = videos["label"].to_numpy(), videos["score"].to_numpy()
    flagged, cleared = s > hi, s < lo
    fakes, reals = y == 1, y == 0

    def share(a: Any, b: Any) -> float:
        return float(a.sum() / b.sum()) if b.sum() else float("nan")

    return {
        "precision": share(flagged & fakes, flagged),
        "recall": share(flagged & fakes, fakes),
        "false_alarm_rate": share(flagged & reals, reals),
        "missed_rate": share(cleared & fakes, fakes),
        "uncertain_rate": float((~flagged & ~cleared).mean()) if len(s) else float("nan"),
    }


def score_sets(ckpt: Path, datasets: tuple[str, ...], split: str = "test") -> dict[str, Any]:
    """Score one checkpoint, with its own calibration, on the given datasets' split."""
    from deepfake_detector.calibrate import Calibration
    from deepfake_detector.data.datasets import FaceCropDataset, load_manifests
    from deepfake_detector.inference import load_detector, predict_frames
    from deepfake_detector.metrics import summarize, video_scores

    model, cfg = load_detector(ckpt)
    cal = Calibration.load(ckpt.parent / "calibration.json")
    how = cfg["train"]["val"]["video_score"]
    root = Path(cfg["data"]["root"])
    device = next(model.parameters()).device
    out: dict[str, Any] = {}
    for name in datasets:
        df = load_manifests(cfg["data"]["manifest_dir"], name, split=split)
        if df.empty:
            continue
        loader = DataLoader(FaceCropDataset(df, root), batch_size=64, shuffle=False)
        frames = predict_frames(model, loader, device, device.type == "cuda")
        frames = frames.assign(prob=cal.apply(frames["logit"].to_numpy()))
        videos = video_scores(frames, how)
        entry: dict[str, Any] = {
            "bands": decision_metrics(videos, cal.likely_real_below, cal.likely_fake_above),
            "items": int(len(videos)),
        }
        if frames["label"].nunique() == 2:
            entry["all"] = summarize(frames, how=how)
            real = frames[frames["label"] == 0]
            kinds = frames[frames["label"] == 1]["method"].unique()
            for m in kinds if len(kinds) > 1 else []:
                fakes = frames[frames["method"] == m]
                entry[str(m)] = summarize(pd.concat([real, fakes]), how=how)
        out[name.split("_f")[0]] = entry
    return out


def finish(r: Recipe, compare_with: tuple[str, ...] = ()) -> dict[str, Any]:
    """Calibrate on the joint validation split, test every dataset, compare with older runs."""
    from deepfake_detector.calibrate import calibrate_frames
    from deepfake_detector.evaluate import score_split
    from deepfake_detector.metrics import summarize

    ckpt = Path("runs") / r.run_name / "best.pt"
    val, cfg = score_split(ckpt, split="val")
    how = cfg["train"]["val"]["video_score"]
    cal = calibrate_frames(val, how)
    cal.fitted_on = "joint val: " + ", ".join(r.datasets)
    cal.save(ckpt.parent / "calibration.json")

    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "temperature": cal.temperature,
        "bands": [cal.likely_real_below, cal.likely_fake_above],
        "val": summarize(val.assign(prob=cal.apply(val["logit"].to_numpy())), how=how),
        "test": score_sets(ckpt, r.datasets),
        "note": OPEN_NOTE,
        "compare": {},
    }
    for other in compare_with:
        other_ckpt = Path("runs") / other / "best.pt"
        if other_ckpt.exists():
            report["compare"][other] = score_sets(other_ckpt, r.datasets)
    (ckpt.parent / "results.json").write_text(json.dumps(report, indent=2))
    (ckpt.parent / "MODEL.md").write_text(model_markdown(report))
    log.info("test: %s", {k: v.get("all", {}).get("video_auc") for k, v in report["test"].items()})
    return report


def _table(results: dict[str, Any]) -> list[str]:
    rows = [
        "| Test set | Fakes | AUC | EER | Precision | Recall | False alarms | Uncertain "
        "| Items |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for name, entry in results.items():
        b = entry["bands"]
        tail = (
            f"{b['precision']:.3f} | {b['recall']:.3f} | {b['false_alarm_rate']:.3f} | "
            f"{b['uncertain_rate']:.3f} | {entry['items']} |"
        )
        if "all" not in entry:  # real photos only: false alarms are the whole story
            rows.append(f"| {name} | (real only) | | | | | {b['false_alarm_rate']:.3f} | "
                        f"{b['uncertain_rate']:.3f} | {entry['items']} |\n")  # fmt: skip
            continue
        m = entry["all"]
        rows.append(f"| {name} | all | {m['video_auc']:.4f} | {m['video_eer']:.4f} | {tail}\n")
        for kind, k in entry.items():
            if kind not in ("all", "bands", "items"):
                rows.append(f"| {name} | {kind} | {k['video_auc']:.4f} | {k['video_eer']:.4f} "
                            "| | | | | |\n")  # fmt: skip
    return rows


def model_markdown(r: dict[str, Any]) -> str:
    lo, hi = r["bands"]
    lines = [
        "# Open-data model\n\n",
        f"{r['note']}\n\n",
        f"Checkpoint `{r['checkpoint']}`; temperature {r['temperature']:.3f}; no manipulation "
        f"detected below {lo:.3f}, likely manipulated above {hi:.3f}.\n\n",
        "Held-out test splits (people or source videos never used in training). For video "
        "sets, one score per face sequence (mean of its frames). Precision, recall, false "
        "alarms and uncertain are read at the bands the app shows: flagged means above the "
        "likely-manipulated cut-off.\n\n",
        *_table(r["test"]),
    ]
    for other, res in r.get("compare", {}).items():
        lines += [f"\nSame test sets, earlier model `{other}`:\n\n", *_table(res)]
    return "".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["all", "fetch", "preprocess", "train", "finish"])
    p.add_argument("--recipe", choices=sorted(RECIPES), default="v1")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--downloads", type=Path, default=Path("data/downloads/openforensics"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--compare", nargs="*", default=[], help="older runs to score alongside")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = RECIPES[args.recipe]
    if args.step in ("all", "fetch"):
        fetch(args.raw, args.downloads, r)
    if args.step in ("all", "preprocess"):
        preprocess(args.raw, args.data, r, args.workers)
    if args.step in ("all", "train"):
        train(r, args.epochs)
    if args.step in ("all", "finish"):
        finish(r, tuple(args.compare))


if __name__ == "__main__":
    main()
