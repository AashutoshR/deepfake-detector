"""Results per demographic group (FR-E8, S-5).

Uses published demographic labels where they exist; this project never infers anyone's
demographics itself. The labels file is a CSV with a video_id column (the same ids as the
manifest, e.g. "Celeb-real/id0_0000" or "Deepfakes/000_003") and one column per attribute,
for example gender, age_group or skin_tone. Leave a cell empty when it is unknown.

For each attribute and group it reports, at video level:
  AUC                       how well real and fake are told apart inside the group
  false alarm rate          share of real videos put in the "likely manipulated" band
  miss rate                 share of fake videos put in the "likely real" band
Groups with fewer than `min_videos` videos, or without both real and fake videos, get no AUC.
Any attribute whose group AUCs differ by more than 5 points is flagged.

    python -m deepfake_detector.fairness --eval-dir outputs/eval/<run>/best \\
        --labels celebdf_v2=data/labels/celebdf_v2.csv ffpp_c23=data/labels/ffpp.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepfake_detector.calibrate import Calibration
from deepfake_detector.metrics import auc, video_scores

log = logging.getLogger(__name__)

GAP_FLAG = 0.05  # 5 AUC points, from the plan


def group_metrics(
    videos: pd.DataFrame,
    labels: pd.DataFrame,
    cutoffs: tuple[float, float],
    min_videos: int = 20,
) -> dict[str, Any]:
    """Per-attribute, per-group metrics for a video table (video_id, label, score)."""
    lo, hi = cutoffs
    merged = videos.merge(labels, on="video_id", how="left")
    attributes = [c for c in labels.columns if c != "video_id"]
    out: dict[str, Any] = {"n_videos": len(videos), "attributes": {}}
    for attr in attributes:
        known = merged[merged[attr].notna() & (merged[attr].astype(str) != "")]
        groups: dict[str, Any] = {}
        for value, g in known.groupby(attr):
            real, fake = g[g["label"] == 0], g[g["label"] == 1]
            enough = len(g) >= min_videos and len(real) > 0 and len(fake) > 0
            groups[str(value)] = {
                "n_videos": len(g),
                "n_real": len(real),
                "n_fake": len(fake),
                "auc": auc(g["label"].to_numpy(), g["score"].to_numpy()) if enough else None,
                "false_alarm_rate": float((real["score"] > hi).mean()) if len(real) else None,
                "miss_rate": float((fake["score"] < lo).mean()) if len(fake) else None,
            }
        aucs = [v["auc"] for v in groups.values() if v["auc"] is not None]
        gap = float(max(aucs) - min(aucs)) if len(aucs) >= 2 else None
        out["attributes"][attr] = {
            "groups": groups,
            "coverage": float(len(known) / len(videos)) if len(videos) else 0.0,
            "auc_gap": gap,
            "flagged": bool(gap is not None and gap > GAP_FLAG),
        }
    return out


def fairness_for_eval(
    eval_dir: Path, labels: dict[str, Path], min_videos: int = 20
) -> dict[str, Any]:
    """Fairness results for each test set that has a labels file, from saved frame scores."""
    results = json.loads((eval_dir / "results.json").read_text())
    cal_path = Path(results["checkpoint"]).parent / "calibration.json"
    cal = Calibration.load(cal_path) if cal_path.exists() else Calibration()
    how = results.get("video_score", "mean")
    out: dict[str, Any] = {
        "cutoffs": [cal.likely_real_below, cal.likely_fake_above],
        "min_videos": min_videos,
        "gap_flag": GAP_FLAG,
        "sets": {},
    }
    for name, path in labels.items():
        frames_path = eval_dir / f"{name}_frames.parquet"
        if not frames_path.exists():
            out["sets"][name] = {"status": "not evaluated"}
            continue
        videos = video_scores(pd.read_parquet(frames_path), how)
        table = pd.read_csv(path, dtype=str)
        if "video_id" not in table.columns:
            raise ValueError(f"{path} has no video_id column")
        res = group_metrics(videos, table, (cal.likely_real_below, cal.likely_fake_above),
                            min_videos)  # fmt: skip
        res["status"] = "ok"
        res["labels_file"] = Path(path).name
        out["sets"][name] = res
    return out


def fairness_markdown(fair: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

    lo, hi = fair["cutoffs"]
    lines = [
        f"Video level. False alarm: real video in the likely-manipulated band (above {hi:.3f}). "
        f"Miss: fake video in the likely-real band (below {lo:.3f}). No AUC for groups under "
        f"{fair['min_videos']} videos. Gaps above {fair['gap_flag'] * 100:.0f} AUC points are "
        "flagged.",
    ]
    for name, res in fair["sets"].items():
        if res.get("status") != "ok":
            lines += ["", f"**{name}**: not evaluated."]
            continue
        for attr, a in res["attributes"].items():
            flag = " **FLAGGED**" if a["flagged"] else ""
            lines += [
                "",
                f"**{name}, {attr}** (labels for {a['coverage'] * 100:.0f}% of videos; "
                f"AUC gap {f(a['auc_gap'])}){flag}",
                "",
                "| Group | Videos (real / fake) | AUC | False alarm rate | Miss rate |",
                "|---|---|---|---|---|",
            ]
            for g, m in a["groups"].items():
                lines.append(
                    f"| {g} | {m['n_videos']} ({m['n_real']} / {m['n_fake']}) | {f(m['auc'])} "
                    f"| {f(m['false_alarm_rate'])} | {f(m['miss_rate'])} |"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--labels", nargs="+", required=True, help="<test set>=<labels.csv>")
    p.add_argument("--min-videos", type=int, default=20)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    labels = {}
    for item in args.labels:
        name, _, path = item.partition("=")
        if not path:
            p.error(f"expected <test set>=<labels.csv>, got {item!r}")
        labels[name] = Path(path)
    fair = fairness_for_eval(args.eval_dir, labels, args.min_videos)
    (args.eval_dir / "fairness.json").write_text(json.dumps(fair, indent=2))
    (args.eval_dir / "fairness.md").write_text("# Fairness\n\n" + fairness_markdown(fair) + "\n")
    flagged = [
        f"{n}/{a}"
        for n, r in fair["sets"].items()
        for a, v in r.get("attributes", {}).items()
        if v["flagged"]
    ]
    log.info("wrote %s; flagged: %s", args.eval_dir / "fairness.md", flagged or "none")


if __name__ == "__main__":
    main()
