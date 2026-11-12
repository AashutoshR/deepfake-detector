"""Test a checkpoint on every configured test set (FR-E1 to FR-E7).

One command writes, per checkpoint:
  results.md / results.json   one row per test set: video AUC with 95% bootstrap CI, frame AUC,
                              EER, accuracy, log loss, ECE before and after calibration
  roc.png                     ROC curves of all test sets on one chart
  calibration_<set>.png       reliability diagrams before and after temperature scaling
  <set>_frames.parquet        every frame score, for later analysis without re-running
  robustness.json / .png      AUC against degradation strength (with --robustness)

Cross-dataset rows sit in the same table, same format, as the same-dataset row (S-2).

    python -m deepfake_detector.evaluate --checkpoint runs/<run>/best.pt
    python -m deepfake_detector.evaluate --checkpoint runs/<run>/best.pt --robustness
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from deepfake_detector.calibrate import Calibration  # noqa: E402
from deepfake_detector.data.datasets import FaceCropDataset, load_manifest  # noqa: E402
from deepfake_detector.data.extract import sample_indices  # noqa: E402
from deepfake_detector.data.perturb import PerturbTransform  # noqa: E402
from deepfake_detector.inference import load_detector, predict_frames  # noqa: E402
from deepfake_detector.metrics import auc, auc_per_method, ece, summarize, video_scores  # noqa: E402

log = logging.getLogger(__name__)

# name -> (manifest folder, split, compressions or None for all)
TEST_SETS: dict[str, tuple[str, str, list[str] | None]] = {
    "ffpp_c23": ("ffpp", "test", ["c23"]),
    "ffpp_c40": ("ffpp", "test", ["c40"]),
    "celebdf_v2": ("celebdf_v2", "test", None),
    "dfdc": ("dfdc", "test", None),
    "deeperforensics": ("deeperforensics", "test", None),
    "wilddeepfake": ("wilddeepfake", "test", None),
    "df40": ("df40", "test", None),
}


def bootstrap_auc_ci(
    labels: np.ndarray, scores: np.ndarray, n: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC, resampling videos with replacement."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2 or n <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2:
            stats.append(auc(labels[idx], scores[idx]))
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def _manifest_path(cfg: dict[str, Any], dataset: str, data_root: Path | None) -> Path:
    base = data_root / "manifests" if data_root else Path(cfg["data"]["manifest_dir"])
    return base / dataset / "manifest.parquet"


def _loader(
    df: pd.DataFrame, root: Path, batch: int, workers: int, transform: Any = None
) -> DataLoader:
    ds = FaceCropDataset(df, root, transform)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def score_split(
    checkpoint: str | Path,
    split: str = "val",
    data_root: Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the training dataset's split with the run's own filters (used for calibration)."""
    model, cfg = load_detector(checkpoint)
    t = cfg["train"]
    root = data_root or Path(cfg["data"]["root"])
    df = load_manifest(
        _manifest_path(cfg, t["dataset"], data_root),
        split=split,
        leave_out_method=t["leave_out_method"],
        compressions=t["compressions"],
    )
    device = next(model.parameters()).device
    frames = predict_frames(
        model, _loader(df, root, batch_size, num_workers), device, bf16=device.type == "cuda"
    )
    return frames, cfg


def evaluate(
    checkpoint: str | Path,
    eval_cfg: dict[str, Any],
    out_dir: Path,
    calibration: Calibration | None = None,
    data_root: Path | None = None,
    robustness: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    model, cfg = load_detector(checkpoint)
    device = next(model.parameters()).device
    bf16 = device.type == "cuda"
    root = data_root or Path(cfg["data"]["root"])
    if calibration is None:
        cal_path = checkpoint.parent / "calibration.json"
        calibration = Calibration.load(cal_path) if cal_path.exists() else Calibration()
        if not cal_path.exists():
            log.warning("no calibration.json found; using temperature 1 (uncalibrated)")
    how, k = eval_cfg["video_score"], eval_cfg["topk"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data_root": str(root),
        "temperature": calibration.temperature,
        "leave_out_method": cfg["train"]["leave_out_method"],
        "video_score": how,
        "sets": {},
    }
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in eval_cfg["test_sets"]:
        dataset, split, comps = TEST_SETS[name]
        path = _manifest_path(cfg, dataset, data_root)
        df = load_manifest(path, split=split, compressions=comps) if path.exists() else None
        if df is None or df.empty:
            results["sets"][name] = {"status": "not available"}
            continue
        frames = predict_frames(
            model, _loader(df, root, eval_cfg["batch_size"], eval_cfg["num_workers"]), device, bf16
        )
        frames["prob_uncalibrated"] = frames["prob"]
        frames["prob"] = calibration.apply(frames["logit"].to_numpy())
        frames["crop_path"] = df["crop_path"].to_numpy()  # loader keeps manifest order
        frames["frame"] = df["frame"].to_numpy()
        frames.to_parquet(out_dir / f"{name}_frames.parquet", index=False)

        m: dict[str, Any] = dict(summarize(frames, how, k))
        raw = video_scores(frames, how, k, score_col="prob_uncalibrated")
        cal = video_scores(frames, how, k)
        m["video_ece_uncalibrated"] = ece(
            raw["label"].to_numpy(), raw["score"].to_numpy(), eval_cfg["ece_bins"]
        )
        lo, hi = bootstrap_auc_ci(
            cal["label"].to_numpy(),
            cal["score"].to_numpy(),
            eval_cfg["bootstrap"]["n"],
            eval_cfg["bootstrap"]["seed"],
        )
        m["video_auc_ci"] = [lo, hi]
        if frames["method"].nunique() > 2:
            m["per_method_auc"] = auc_per_method(frames, how)
        m["status"] = "ok"
        results["sets"][name] = m
        if cal["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(cal["label"], cal["score"])
            roc_data[name] = (fpr, tpr)
        _plot_reliability(raw, cal, name, out_dir / f"calibration_{name}.png", eval_cfg["ece_bins"])
        log.info("%s: video AUC %.4f [%.4f, %.4f]", name, m["video_auc"], lo, hi)

    _plot_roc(roc_data, results, out_dir / "roc.png")
    if robustness:
        results["robustness"] = _robustness(
            model, cfg, eval_cfg, root, data_root, calibration, device, out_dir
        )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(results_markdown(results))
    return results


def _robustness(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    root: Path,
    data_root: Path | None,
    calibration: Calibration,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    r = eval_cfg["robustness"]
    dataset, split, comps = TEST_SETS[r["test_set"]]
    path = _manifest_path(cfg, dataset, data_root)
    if not path.exists():
        return {"status": "not available"}
    df = load_manifest(path, split=split, compressions=comps)
    parts = [
        g.iloc[sample_indices(len(g), r["frames_per_video"])]
        for _, g in df.groupby("video_id", sort=False)
    ]
    df = pd.concat(parts, ignore_index=True)

    def video_auc(transform: Any) -> float:
        loader = _loader(df, root, eval_cfg["batch_size"], eval_cfg["num_workers"], transform)
        frames = predict_frames(model, loader, device, device.type == "cuda")
        frames["prob"] = calibration.apply(frames["logit"].to_numpy())
        return summarize(frames, eval_cfg["video_score"], eval_cfg["topk"])["video_auc"]

    out: dict[str, Any] = {"test_set": r["test_set"], "clean": video_auc(None), "sweeps": {}}
    for kind, strengths in r["sweeps"].items():
        out["sweeps"][kind] = {str(s): video_auc(PerturbTransform(kind, s)) for s in strengths}
        log.info("robustness %s: %s", kind, out["sweeps"][kind])
    (out_dir / "robustness.json").write_text(json.dumps(out, indent=2))
    _plot_robustness(out, out_dir / "robustness.png")
    return out


# ----------------------------------------------------------------------------- report


def _fmt(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def results_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Results",
        "",
        f"Checkpoint: `{results['checkpoint']}`  ",
        f"Video score: {results['video_score']} of frame scores; temperature "
        f"{results['temperature']:.3f} (fitted on FF++ validation)  ",
        f"Held-out method in training: {results['leave_out_method'] or 'none'}",
        "",
        "| Test set | Videos | Video AUC (95% CI) | Frame AUC | EER | Accuracy | Log loss "
        "| ECE before | ECE after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in results["sets"].items():
        if m.get("status") != "ok":
            lines.append(f"| {name} | not available | | | | | | | |")
            continue
        lo, hi = m["video_auc_ci"]
        lines.append(
            f"| {name} | {int(m['n_videos'])} | {_fmt(m['video_auc'])} ({_fmt(lo)} to {_fmt(hi)}) "
            f"| {_fmt(m['frame_auc'])} | {_fmt(m['video_eer'])} | {_fmt(m['video_acc'])} "
            f"| {_fmt(m['video_log_loss'])} | {_fmt(m['video_ece_uncalibrated'])} "
            f"| {_fmt(m['video_ece'])} |"
        )
    for name, m in results["sets"].items():
        if "per_method_auc" in m:
            lines += [
                "",
                f"## Video AUC per fake method ({name})",
                "",
                "| Method | AUC |",
                "|---|---|",
            ]
            lines += [f"| {k} | {_fmt(v)} |" for k, v in m["per_method_auc"].items()]
    rob = results.get("robustness")
    if rob and "sweeps" in rob:
        lines += [
            "",
            f"## Robustness ({rob['test_set']}, clean AUC {_fmt(rob['clean'])})",
            "",
            "| Degradation | Strength | Video AUC |",
            "|---|---|---|",
        ]
        for kind, vals in rob["sweeps"].items():
            lines += [f"| {kind} | {s} | {_fmt(v)} |" for s, v in vals.items()]
    lines += [
        "",
        "Scores are estimates from a statistical model, not proof that a video is real "
        "or manipulated.",
    ]
    return "\n".join(lines) + "\n"


def _plot_roc(roc: dict[str, tuple[np.ndarray, np.ndarray]], results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr) in roc.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {results['sets'][name]['video_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Video-level ROC")
    if roc:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reliability(labels: np.ndarray, probs: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    conf = np.array([probs[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    acc = np.array([labels[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    return conf, acc


def _plot_reliability(
    raw: pd.DataFrame, cal: pd.DataFrame, name: str, path: Path, bins: int
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for df, label in [(raw, "before"), (cal, "after")]:
        y, p = df["label"].to_numpy(float), df["score"].to_numpy(float)
        conf, acc = _reliability(y, p, bins)
        ax.plot(conf, acc, "o-", label=f"{label} (ECE {ece(y, p, bins):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(
        xlabel="Predicted fake probability",
        ylabel="Fraction fake",
        title=f"Calibration: {name}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_robustness(rob: dict[str, Any], path: Path) -> None:
    sweeps = rob["sweeps"]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(3.2 * len(sweeps), 3), squeeze=False)
    for ax, (kind, vals) in zip(axes[0], sweeps.items(), strict=False):
        xs = [float(s) for s in vals]
        ax.plot(xs, list(vals.values()), "o-")
        ax.axhline(rob["clean"], ls="--", color="grey", lw=1)
        ax.set(title=kind, xlabel="strength", ylabel="video AUC", ylim=(0.4, 1.0))
        if kind in ("jpeg", "downscale", "contrast"):
            ax.invert_xaxis()  # left to right = getting worse
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--sets", nargs="+", default=None, help="subset of test sets")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--robustness", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loaded = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert isinstance(loaded, dict)
    eval_cfg: dict[str, Any] = {str(k): v for k, v in loaded.items()}
    if args.sets:
        eval_cfg["test_sets"] = args.sets
    out = args.out or Path(eval_cfg["out_dir"]) / args.checkpoint.parent.name / args.checkpoint.stem
    evaluate(args.checkpoint, eval_cfg, out, data_root=args.data_root, robustness=args.robustness)
    log.info("wrote %s", out / "results.md")


if __name__ == "__main__":
    main()
