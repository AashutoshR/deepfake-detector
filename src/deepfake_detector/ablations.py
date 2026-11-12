"""Ablation grid A1 to A8: one launcher, one results table (FR-E9).

Each ablation changes one setting of the final model and is scored with 1 seed on
  M1  FF++ c23 test video AUC
  M2  Celeb-DF v2 test video AUC
  M5  mean video AUC on the held-out FF++ method, one leave-one-method-out run per method

Options that equal the base config share one run, so the base model is trained once. A8 (how
frame scores become a video score) needs no training: it re-scores the saved frame scores.
Runs are resumable: finished ones are skipped, stopped ones resume from last.pt.

    python scripts/run_ablations.py --dry-run                 # list runs and time estimate
    python scripts/run_ablations.py --only A1 A3 --epochs 8
    python scripts/run_ablations.py --m5-methods NeuralTextures   # cheaper M5
    python scripts/run_ablations.py --gather-only             # rebuild the table
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from deepfake_detector.data.indexers import FFPP_METHODS
from deepfake_detector.metrics import auc_per_method, summarize
from deepfake_detector.train import load_config

log = logging.getLogger(__name__)

# ablation id -> (what changes, {option label: overrides})
ABLATIONS: dict[str, tuple[str, dict[str, list[str]]]] = {
    "A1": (
        "Starting weights",
        {
            "ImageNet-21k": ["model.pretrained=imagenet21k"],
            "CLIP": ["model.pretrained=clip"],
            "DINOv2": ["model.pretrained=dinov2"],
        },
    ),
    "A2": (
        "Fine-tuning",
        {
            "LoRA": ["model.finetune.mode=lora"],
            "Full": ["model.finetune.mode=full", "train.optimizer.lr=3.0e-05"],
            "Last layer only": ["model.finetune.mode=last_layer"],
        },
    ),
    "A3": (
        "Synthetic fakes",
        {
            "None": ["train.sbi.enabled=false"],
            "Self-blended only": ["train.sbi.only=true"],
            "Self-blended plus FF++ fakes": ["train.sbi.enabled=true", "train.sbi.only=false"],
        },
    ),
    "A4": (
        "Augmentation",
        {"Basic": ["train.augment.preset=basic"], "Full": ["train.augment.preset=full"]},
    ),
    "A5": (
        "Token pooling",
        {"CLS token": ["model.pooling=cls"], "Mean of patch tokens": ["model.pooling=mean_patch"]},
    ),
    "A6": (
        "Extra head",
        {
            "Real/fake only": ["model.method_head.enabled=false"],
            "Also predict method": ["model.method_head.enabled=true"],
        },
    ),
    "A7": (
        "Training compression",
        {"c23 only": ["train.compressions=[c23]"], "c23 and c40": ["train.compressions=[c23,c40]"]},
    ),
    "A8": ("Video score", {"Mean": [], "Median": [], "Top-k": []}),
}
A8_HOW = {"Mean": "mean", "Median": "median", "Top-k": "topk"}
HOURS_PER_20_EPOCHS = 3.5  # measured speed on the RTX 4060 Laptop, see README


@dataclass
class Run:
    name: str
    overrides: list[str]
    sets: list[str]
    used_by: list[str] = field(default_factory=list)


def effective_overrides(base: DictConfig, overrides: Sequence[str]) -> list[str]:
    """Drop overrides that only restate the base config, so equal options share a run."""
    out = []
    for o in overrides:
        key = o.split("=", 1)[0]
        new = OmegaConf.select(OmegaConf.from_dotlist([o]), key)
        if OmegaConf.select(base, key) != new:
            out.append(o)
    return sorted(out)


def run_name(overrides: Sequence[str], leave_out: str | None = None) -> str:
    def slug(o: str) -> str:
        key, value = o.split("=", 1)
        clean = "".join(ch if ch.isalnum() or ch in ".-" else "" for ch in value.replace(",", "-"))
        # Keep everything after the section name so "train.sbi.enabled" and
        # "model.method_head.enabled" never produce the same run name.
        return f"{key.split('.', 1)[-1]}-{clean}"

    name = "abl_" + ("_".join(slug(o) for o in overrides) if overrides else "base")
    return f"{name}__lomo-{leave_out}" if leave_out else name


def plan(
    only: Sequence[str] | None = None,
    m5_methods: Sequence[str] = FFPP_METHODS,
    base: DictConfig | None = None,
) -> dict[str, Run]:
    """All runs needed for the chosen ablations, deduplicated."""
    base = base if base is not None else load_config()
    runs: dict[str, Run] = {}

    def add(overrides: list[str], leave_out: str | None, sets: list[str], user: str) -> None:
        extra = [f"train.leave_out_method={leave_out}"] if leave_out else []
        name = run_name(overrides, leave_out)
        run = runs.setdefault(name, Run(name, overrides + extra, sets))
        run.used_by.append(user)

    for abl in only or list(ABLATIONS):
        _, options = ABLATIONS[abl]
        for label, raw in options.items():
            eff = effective_overrides(base, raw)
            add(eff, None, ["ffpp_c23", "celebdf_v2"], f"{abl} {label}")
            for m in m5_methods:
                add(eff, m, ["ffpp_c23"], f"{abl} {label} (M5)")
    return runs


Runner = Callable[[list[str]], None]


def _subprocess_runner(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def execute(
    runs: dict[str, Run],
    runs_dir: Path,
    out_dir: Path,
    eval_config: Path,
    common: Sequence[str] = (),
    runner: Runner = _subprocess_runner,
) -> None:
    """Train and test every run that has no results yet; resume stopped runs."""
    for i, run in enumerate(runs.values(), 1):
        result = out_dir / run.name / "results.json"
        if result.exists():
            log.info("[%d/%d] %s: done, skipping", i, len(runs), run.name)
            continue
        log.info("[%d/%d] %s: %s", i, len(runs), run.name, " ".join(run.overrides) or "base")
        last = runs_dir / run.name / "last.pt"
        if last.exists():
            runner(["deepfake_detector.train", "--resume", str(last)])
        else:
            names = [f"train.out_dir={runs_dir.as_posix()}", f"train.run_name={run.name}"]
            runner(["deepfake_detector.train", *common, *run.overrides, *names])
        runner(
            [
                "deepfake_detector.evaluate",
                "--checkpoint", str(runs_dir / run.name / "best.pt"),
                "--config", str(eval_config),
                "--out", str(out_dir / run.name),
                "--sets", *run.sets,
            ]
        )  # fmt: skip


def _load(out_dir: Path, name: str) -> dict[str, Any] | None:
    path = out_dir / name / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def _auc(res: dict[str, Any] | None, test_set: str) -> float:
    if res is None:
        return float("nan")
    s = res["sets"].get(test_set, {})
    return float(s["video_auc"]) if s.get("status") == "ok" else float("nan")


def _rescore(out_dir: Path, name: str, test_set: str, how: str) -> pd.DataFrame | None:
    path = out_dir / name / f"{test_set}_frames.parquet"
    return pd.read_parquet(path) if path.exists() else None


def gather(
    out_dir: Path,
    only: Sequence[str] | None = None,
    m5_methods: Sequence[str] = FFPP_METHODS,
    base: DictConfig | None = None,
) -> pd.DataFrame:
    """One row per ablation option with M1, M2 and M5 (NaN where not run yet)."""
    base = base if base is not None else load_config()
    rows = []
    for abl in only or list(ABLATIONS):
        what, options = ABLATIONS[abl]
        for label, raw in options.items():
            eff = effective_overrides(base, raw)
            name = run_name(eff)
            if abl == "A8":
                how = A8_HOW[label]
                m1 = m2 = float("nan")
                f1 = _rescore(out_dir, name, "ffpp_c23", how)
                f2 = _rescore(out_dir, name, "celebdf_v2", how)
                if f1 is not None:
                    m1 = summarize(f1, how)["video_auc"]  # type: ignore[arg-type]
                if f2 is not None:
                    m2 = summarize(f2, how)["video_auc"]  # type: ignore[arg-type]
                held = []
                for m in m5_methods:
                    f = _rescore(out_dir, run_name(eff, m), "ffpp_c23", how)
                    held.append(auc_per_method(f, how).get(m, np.nan) if f is not None else np.nan)  # type: ignore[arg-type]
            else:
                res = _load(out_dir, name)
                m1, m2 = _auc(res, "ffpp_c23"), _auc(res, "celebdf_v2")
                held = []
                for m in m5_methods:
                    r = _load(out_dir, run_name(eff, m))
                    per = (r or {}).get("sets", {}).get("ffpp_c23", {}).get("per_method_auc", {})
                    held.append(per.get(m, np.nan))
            done = [h for h in held if not np.isnan(h)]
            m5 = float(np.mean(done)) if len(done) == len(m5_methods) else float("nan")
            rows.append(
                {"ablation": abl, "what": what, "option": label, "run": name,
                 "M1": m1, "M2": m2, "M5": m5, "M5_runs_done": f"{len(done)}/{len(m5_methods)}"}
            )  # fmt: skip
    return pd.DataFrame(rows)


def table_markdown(df: pd.DataFrame, epochs: int | None, m5_methods: Sequence[str]) -> str:
    def f(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    lines = [
        "# Ablation results",
        "",
        f"1 seed{f', {epochs} epochs' if epochs else ''}. M1: FF++ c23 test video AUC. "
        f"M2: Celeb-DF v2 test video AUC. M5: mean held-out-method AUC over "
        f"{', '.join(m5_methods)}. Best value per ablation in bold.",
        "",
        "| # | What changes | Option | M1 | M2 | M5 |",
        "|---|---|---|---|---|---|",
    ]
    for abl, part in df.groupby("ablation", sort=False):
        best = {m: part[m].max() for m in ("M1", "M2", "M5")}
        for r in part.itertuples(index=False):
            cells = []
            for m in ("M1", "M2", "M5"):
                v = getattr(r, m)
                cells.append(f"**{f(v)}**" if not pd.isna(v) and v == best[m] else f(v))
            lines.append(f"| {abl} | {r.what} | {r.option} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, runner: Runner = _subprocess_runner) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(ABLATIONS), default=None)
    p.add_argument("--epochs", type=int, default=8, help="shorter schedule for ablations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--m5-methods", nargs="+", choices=list(FFPP_METHODS), default=list(FFPP_METHODS)
    )
    p.add_argument("--runs-dir", type=Path, default=Path("runs/ablations"))
    p.add_argument("--out", type=Path, default=Path("outputs/ablations"))
    p.add_argument("--eval-config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--bootstrap", type=int, default=200, help="bootstrap samples per test set")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gather-only", action="store_true")
    p.add_argument(
        "overrides", nargs="*", help="extra overrides for every run, e.g. data.root=D:/data"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = load_config(overrides=args.overrides)
    runs = plan(args.only, args.m5_methods, base)
    todo = [r for r in runs.values() if not (args.out / r.name / "results.json").exists()]
    hours = len(todo) * HOURS_PER_20_EPOCHS * args.epochs / 20
    log.info("%d runs needed, %d not done yet; about %.0f GPU hours left at %d epochs",
             len(runs), len(todo), hours, args.epochs)  # fmt: skip

    if args.dry_run:
        for r in runs.values():
            state = "done" if r not in todo else "todo"
            print(f"{state:5} {r.name:60} {' '.join(r.overrides) or '(base config)'}")
        return pd.DataFrame()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.gather_only:
        eval_cfg = yaml.safe_load(args.eval_config.read_text())
        eval_cfg["bootstrap"]["n"] = args.bootstrap
        eval_path = args.out / "eval.yaml"
        eval_path.write_text(yaml.safe_dump(eval_cfg))
        common = [f"train.epochs={args.epochs}", f"train.seed={args.seed}", *args.overrides]
        execute(runs, args.runs_dir, args.out, eval_path, common, runner)

    df = gather(args.out, args.only, args.m5_methods, base)
    df.to_json(args.out / "ablations.json", orient="records", indent=2)
    (args.out / "ablations.md").write_text(table_markdown(df, args.epochs, args.m5_methods))
    log.info("wrote %s", args.out / "ablations.md")
    return df


if __name__ == "__main__":
    main()
