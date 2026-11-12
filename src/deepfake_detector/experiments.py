"""Headline experiments: the final model and the three baselines over 3 seeds.

For every model and seed this trains, calibrates and tests a main run on all test sets, plus one
leave-one-method-out run per FF++ method for M5. It then reports mean and standard deviation over
seeds for every target in the plan, checks the targets, and checks that the final model beats
every baseline on M2 and M5.

Baselines use the same data pipeline as the final model, so differences come from the model:
  plain_vit        ViT-B/16, full fine-tuning, no self-blended images, basic augmentation
  xception         Xception, full fine-tuning, same pipeline as the final model
  efficientnet_b4  EfficientNet-B4, full fine-tuning, same pipeline as the final model

    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --only final --seeds 0 1 2
    python scripts/run_experiments.py --gather-only
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from deepfake_detector.ablations import Run, Runner, _load, _subprocess_runner, execute
from deepfake_detector.data.indexers import FFPP_METHODS

log = logging.getLogger(__name__)

EXPERIMENTS: dict[str, tuple[str, list[str]]] = {
    "final": ("ViT-B/16 + LoRA + SBI (final)", []),
    "plain_vit": (
        "ViT-B/16, plain fine-tuning",
        [
            "model.finetune.mode=full",
            "train.optimizer.lr=3.0e-05",
            "train.sbi.enabled=false",
            "train.augment.preset=basic",
        ],
    ),
    "xception": ("Xception", ["model.arch=xception"]),
    "efficientnet_b4": ("EfficientNet-B4", ["model.arch=efficientnet_b4"]),
}
BASELINES = ("plain_vit", "xception", "efficientnet_b4")
# Training images per second on the RTX 4060 Laptop (scripts/bench_gpu.py); final model = 185.
SPEED = {"final": 185, "plain_vit": 97, "xception": 168, "efficientnet_b4": 74}
HOURS_FINAL_20_EPOCHS = 3.5

# id -> (description, test set, metric, target, stretch, higher is better)
TARGETS: dict[str, tuple[str, str, str, float | None, float, bool]] = {
    "M1": ("Same dataset", "ffpp_c23", "video_auc", 0.97, 0.99, True),
    "M2": ("Cross-dataset (headline)", "celebdf_v2", "video_auc", 0.80, 0.88, True),
    "M3": ("Cross-dataset", "dfdc", "video_auc", 0.70, 0.75, True),
    "M4a": ("Cross-dataset", "deeperforensics", "video_auc", None, 0.75, True),
    "M4b": ("Cross-dataset", "wilddeepfake", "video_auc", None, 0.75, True),
    "M5": ("Unseen method (mean of 4)", "ffpp_c23", "held_out_auc", 0.85, 0.90, True),
    "M6": ("Heavier compression", "ffpp_c40", "video_auc", 0.85, 0.90, True),
    "M7": ("Calibration error (ECE)", "celebdf_v2", "video_ece", 0.10, 0.05, False),
}


def run_name(exp: str, seed: int, leave_out: str | None = None) -> str:
    name = f"exp_{exp}_s{seed}"
    return f"{name}__lomo-{leave_out}" if leave_out else name


def plan(
    only: Sequence[str] | None,
    seeds: Sequence[int],
    m5_methods: Sequence[str],
    test_sets: Sequence[str],
) -> dict[str, Run]:
    runs: dict[str, Run] = {}
    for exp in only or list(EXPERIMENTS):
        overrides = EXPERIMENTS[exp][1]
        for seed in seeds:
            s = [f"train.seed={seed}"]
            name = run_name(exp, seed)
            runs[name] = Run(name, [*overrides, *s], list(test_sets), [exp])
            for m in m5_methods:
                lomo = run_name(exp, seed, m)
                extra = [f"train.leave_out_method={m}"]
                runs[lomo] = Run(lomo, [*overrides, *s, *extra], ["ffpp_c23"], [exp])
    return runs


def estimate_hours(runs: dict[str, Run], epochs: dict[str, int]) -> float:
    total = 0.0
    for run in runs.values():
        exp = run.used_by[0]
        total += HOURS_FINAL_20_EPOCHS * epochs[exp] / 20 * SPEED["final"] / SPEED[exp]
    return total


def _seed_metrics(out: Path, exp: str, seed: int, m5_methods: Sequence[str]) -> dict[str, float]:
    res = _load(out, run_name(exp, seed))
    values: dict[str, float] = {}
    for tid, (_, test_set, metric, *_rest) in TARGETS.items():
        if metric == "held_out_auc":
            held = []
            for m in m5_methods:
                r = _load(out, run_name(exp, seed, m))
                per = (r or {}).get("sets", {}).get("ffpp_c23", {}).get("per_method_auc", {})
                held.append(per.get(m, np.nan))
            values[tid] = float(np.mean(held)) if not np.isnan(held).any() else np.nan
        else:
            s = (res or {}).get("sets", {}).get(test_set, {})
            values[tid] = float(s[metric]) if s.get("status") == "ok" else np.nan
    return values


def gather(
    out: Path, only: Sequence[str] | None, seeds: Sequence[int], m5_methods: Sequence[str]
) -> pd.DataFrame:
    """One row per experiment and target: mean, std and number of finished seeds."""
    rows = []
    for exp in only or list(EXPERIMENTS):
        per_seed = [_seed_metrics(out, exp, s, m5_methods) for s in seeds]
        for tid in TARGETS:
            vals = np.array([p[tid] for p in per_seed], dtype=float)
            done = vals[~np.isnan(vals)]
            rows.append(
                {
                    "experiment": exp,
                    "target": tid,
                    "mean": float(done.mean()) if len(done) else np.nan,
                    "std": float(done.std(ddof=1)) if len(done) > 1 else np.nan,
                    "seeds_done": len(done),
                    "per_seed": [None if np.isnan(v) else float(v) for v in vals],
                }
            )
    return pd.DataFrame(rows)


def verdicts(df: pd.DataFrame, n_seeds: int) -> dict[str, Any]:
    """Targets met by the final model, and whether it beats every baseline on M2 and M5."""
    table = df.set_index(["experiment", "target"])
    out: dict[str, Any] = {"targets": {}, "beats_baselines": {}}
    if "final" not in df["experiment"].values:
        return out
    for tid, (_, _, _, target, stretch, higher) in TARGETS.items():
        row = table.loc[("final", tid)]
        mean, complete = row["mean"], row["seeds_done"] == n_seeds
        if target is None or np.isnan(mean):
            state = "report only" if target is None and not np.isnan(mean) else "not run"
        else:
            met = mean >= target if higher else mean <= target
            great = mean >= stretch if higher else mean <= stretch
            state = "stretch met" if great else "met" if met else "missed"
            if not complete:
                state += " (incomplete seeds)"
        out["targets"][tid] = state
    for tid in ("M2", "M5"):
        final = table.loc[("final", tid), "mean"]
        for b in BASELINES:
            if b not in df["experiment"].values:
                continue
            base = table.loc[(b, tid), "mean"]
            out["beats_baselines"][f"{tid} vs {b}"] = (
                None if np.isnan(final) or np.isnan(base) else bool(final > base)
            )
    return out


def results_markdown(df: pd.DataFrame, verdict: dict[str, Any], seeds: Sequence[int]) -> str:
    def cell(exp: str, tid: str) -> str:
        r = df[(df.experiment == exp) & (df.target == tid)]
        if r.empty or np.isnan(r["mean"].iloc[0]):
            return "n/a"
        mean, std, n = r["mean"].iloc[0], r["std"].iloc[0], r["seeds_done"].iloc[0]
        sd = "" if np.isnan(std) else f" ± {std:.3f}"
        partial = "" if n == len(seeds) else f" ({n}/{len(seeds)} seeds)"
        return f"{mean:.3f}{sd}{partial}"

    exps = list(dict.fromkeys(df["experiment"]))
    head = "| Target | Test | Goal | " + " | ".join(EXPERIMENTS[e][0] for e in exps) + " |"
    lines = [
        "# Headline results",
        "",
        f"Mean ± standard deviation over seeds {', '.join(map(str, seeds))}. All models trained "
        "on FF++ c23 only. Video-level AUC unless stated; M7 is calibration error (lower is "
        "better).",
        "",
        head,
        "|" + "---|" * (3 + len(exps)),
    ]
    for tid, (desc, test_set, _, target, stretch, higher) in TARGETS.items():
        sign = ">=" if higher else "<="
        goal = f"{sign} {target}" if target is not None else "report"
        cells = " | ".join(cell(e, tid) for e in exps)
        lines.append(f"| {tid} {desc} | {test_set} | {goal} (stretch {stretch}) | {cells} |")
    if verdict["targets"]:
        lines += ["", "## Final model against the targets", ""]
        lines += [f"- {k}: {v}" for k, v in verdict["targets"].items()]
    if verdict["beats_baselines"]:
        lines += ["", "## Final model against the baselines (must win on M2 and M5)", ""]
        for k, v in verdict["beats_baselines"].items():
            lines.append(f"- {k}: {'yes' if v else 'no' if v is False else 'not run'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, runner: Runner = _subprocess_runner) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20, help="final model epochs")
    p.add_argument("--baseline-epochs", type=int, default=10, help="shorter, as in the plan")
    p.add_argument(
        "--m5-methods", nargs="+", choices=list(FFPP_METHODS), default=list(FFPP_METHODS)
    )
    p.add_argument("--runs-dir", type=Path, default=Path("runs/experiments"))
    p.add_argument("--out", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--eval-config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gather-only", action="store_true")
    p.add_argument("overrides", nargs="*", help="extra overrides for every run")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eval_cfg = yaml.safe_load(args.eval_config.read_text())
    runs = plan(args.only, args.seeds, args.m5_methods, eval_cfg["test_sets"])
    epochs = {e: args.epochs if e == "final" else args.baseline_epochs for e in EXPERIMENTS}
    todo = {k: r for k, r in runs.items() if not (args.out / k / "results.json").exists()}
    log.info("%d runs, %d not done; about %.0f GPU hours left",
             len(runs), len(todo), estimate_hours(todo, epochs))  # fmt: skip
    if args.dry_run:
        for name, r in runs.items():
            state = "todo" if name in todo else "done"
            print(f"{state:5} {name:45} {' '.join(r.overrides)}")
        return pd.DataFrame()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.gather_only:
        eval_path = args.out / "eval.yaml"
        eval_path.write_text(yaml.safe_dump(eval_cfg))
        for exp in args.only or list(EXPERIMENTS):
            exp_runs = {k: r for k, r in runs.items() if r.used_by[0] == exp}
            common = [f"train.epochs={epochs[exp]}", *args.overrides]
            execute(exp_runs, args.runs_dir, args.out, eval_path, common, runner, calibrate=True)

    df = gather(args.out, args.only, args.seeds, args.m5_methods)
    verdict = verdicts(df, len(args.seeds))
    df.to_json(args.out / "results.json", orient="records", indent=2)
    (args.out / "results.md").write_text(
        results_markdown(df, verdict, args.seeds), encoding="utf-8"
    )
    log.info("wrote %s", args.out / "results.md")
    return df


if __name__ == "__main__":
    main()
