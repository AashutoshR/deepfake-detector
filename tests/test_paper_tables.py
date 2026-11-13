import json
from pathlib import Path

from deepfake_detector import paper_tables
from deepfake_detector.paper_tables import ablation_table, main_table


def test_main_table_bolds_best_and_handles_missing() -> None:
    rows = [
        {"experiment": "final", "target": "M2", "mean": 0.85, "std": 0.01},
        {"experiment": "xception", "target": "M2", "mean": 0.80, "std": 0.02},
        {"experiment": "final", "target": "M7", "mean": 0.06, "std": None},
        {"experiment": "xception", "target": "M7", "mean": 0.12, "std": 0.01},
    ]
    tex = main_table(rows)
    assert "Celeb-DF v2 & \\textbf{0.850$\\pm$0.010} & 0.800$\\pm$0.020 \\\\" in tex
    assert "\\textbf{0.060}" in tex  # lower ECE is better
    assert "DFDC subset & n/a & n/a" in tex
    assert tex.count("\\begin{tabular}{lcc}") == 1


def test_ablation_table_groups_rows() -> None:
    rows = [
        {"ablation": "A1", "option": "ImageNet-21k", "M1": 0.98, "M2": 0.81, "M5": 0.86},
        {"ablation": "A1", "option": "CLIP", "M1": 0.97, "M2": float("nan"), "M5": 0.88},
        {"ablation": "A4", "option": "Basic", "M1": 0.99, "M2": 0.70, "M5": 0.80},
    ]
    tex = ablation_table(rows)
    assert "A1 & ImageNet-21k & 0.980 & 0.810 & 0.860" in tex
    assert " & CLIP & 0.970 & n/a & 0.880" in tex
    assert tex.count("\\midrule") == 2
    assert "not run yet" in ablation_table([])


def test_cli_writes_placeholder_tables(tmp_path: Path) -> None:
    out = tmp_path / "tables"
    paper_tables.main(
        [
            "--experiments",
            str(tmp_path / "none.json"),
            "--ablations",
            str(tmp_path / "none.json"),
            "--out",
            str(out),
        ]
    )
    assert "Ours & Plain ViT & Xception & EffNet-B4" in (out / "main.tex").read_text()
    exp = tmp_path / "exp.json"
    exp.write_text(
        json.dumps([{"experiment": "final", "target": "M1", "mean": 0.99, "std": 0.001}])
    )
    paper_tables.main(
        ["--experiments", str(exp), "--ablations", str(tmp_path / "x.json"), "--out", str(out)]
    )
    assert "\\textbf{0.990$\\pm$0.001}" in (out / "main.tex").read_text()
