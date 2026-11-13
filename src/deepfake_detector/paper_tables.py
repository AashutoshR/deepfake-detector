"""Write the paper's LaTeX tables from the experiment and ablation results.

Numbers in the paper come only from these files, never typed by hand:
  outputs/experiments/results.json  ->  paper/tables/main.tex
  outputs/ablations/ablations.json  ->  paper/tables/ablations.tex
Missing results show as "n/a", so the paper always compiles.

    python scripts/paper_tables.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from deepfake_detector.experiments import EXPERIMENTS, TARGETS

ROW_LABELS = {
    "M1": "FF++ c23 (same dataset)",
    "M6": "FF++ c40",
    "M2": "Celeb-DF v2",
    "M3": "DFDC subset",
    "M4a": "DeeperForensics-1.0",
    "M4b": "WildDeepfake",
    "M5": "Unseen FF++ method (mean)",
    "M7": "ECE on Celeb-DF $\\downarrow$",
}
SHORT_NAMES = {
    "final": "Ours",
    "plain_vit": "Plain ViT",
    "xception": "Xception",
    "efficientnet_b4": "EffNet-B4",
}


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def main_table(rows: list[dict[str, Any]]) -> str:
    """Targets as rows, models as columns; best mean per row in bold."""
    exps = [e for e in EXPERIMENTS if any(r["experiment"] == e for r in rows)] or list(EXPERIMENTS)
    lookup = {(r["experiment"], r["target"]): r for r in rows}
    lines = [
        "\\begin{tabular}{l" + "c" * len(exps) + "}",
        "\\toprule",
        "Test set & " + " & ".join(SHORT_NAMES[e] for e in exps) + " \\\\",
        "\\midrule",
    ]
    for tid in ROW_LABELS:
        higher = TARGETS[tid][5]
        means = [lookup.get((e, tid), {}).get("mean") for e in exps]
        valid = [float(m) for m in means if m is not None and not _nan(m)]
        best = (max(valid) if higher else min(valid)) if valid else None
        cells = []
        for e, mean in zip(exps, means, strict=True):
            if _nan(mean):
                cells.append("n/a")
                continue
            std = lookup[(e, tid)].get("std")
            text = f"{mean:.3f}" + ("" if _nan(std) else f"$\\pm${std:.3f}")
            cells.append(f"\\textbf{{{text}}}" if mean == best else text)
        if tid == "M5":
            lines.append("\\midrule")
        lines.append(f"{ROW_LABELS[tid]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "& Option & FF++ & Celeb-DF & Unseen \\\\",
        "\\midrule",
    ]
    if not rows:
        lines.append("\\multicolumn{5}{c}{n/a: ablations not run yet} \\\\")
    previous = None
    for r in rows:
        if previous is not None and r["ablation"] != previous:
            lines.append("\\midrule")
        label = r["ablation"] if r["ablation"] != previous else ""
        previous = r["ablation"]
        cells = ["n/a" if _nan(r[m]) else f"{r[m]:.3f}" for m in ("M1", "M2", "M5")]
        lines.append(f"{label} & {r['option']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _read(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiments", type=Path, default=Path("outputs/experiments/results.json"))
    p.add_argument("--ablations", type=Path, default=Path("outputs/ablations/ablations.json"))
    p.add_argument("--out", type=Path, default=Path("paper/tables"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "main.tex").write_text(main_table(_read(args.experiments)))
    (args.out / "ablations.tex").write_text(ablation_table(_read(args.ablations)))
    print(f"wrote {args.out / 'main.tex'} and {args.out / 'ablations.tex'}")


if __name__ == "__main__":
    main()
