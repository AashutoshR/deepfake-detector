import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepfake_detector import fairness
from deepfake_detector.calibrate import Calibration
from deepfake_detector.fairness import fairness_markdown, group_metrics
from deepfake_detector.release import fill_card


def videos_and_labels(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group A is easy to score, group B is hard: B's fakes look like reals."""
    rng = np.random.default_rng(seed)
    base_score = {("A", 1): 0.85, ("A", 0): 0.15, ("B", 1): 0.5, ("B", 0): 0.45}
    rows, labels = [], []
    for i in range(n):
        group = "A" if i % 2 == 0 else "B"
        label = int(i % 4 < 2)
        score = base_score[(group, label)]
        rows.append({"video_id": f"v{i}", "label": label, "score": score + rng.normal(0, 0.05)})
        labels.append({"video_id": f"v{i}", "skin_tone": group, "gender": "" if i < 10 else "x"})
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_group_metrics_flags_gap_and_counts() -> None:
    videos, labels = videos_and_labels()
    res = group_metrics(videos, labels, cutoffs=(0.3, 0.7), min_videos=20)
    tone = res["attributes"]["skin_tone"]
    a, b = tone["groups"]["A"], tone["groups"]["B"]
    assert a["auc"] > 0.99 and b["auc"] < 0.9
    assert tone["flagged"] and tone["auc_gap"] > 0.05
    assert a["n_videos"] == 100 and a["n_real"] + a["n_fake"] == 100
    assert a["false_alarm_rate"] == 0.0 and b["miss_rate"] == 0.0  # B fakes sit at 0.5
    gender = res["attributes"]["gender"]
    assert gender["coverage"] == pytest.approx(0.95)  # empty cells are unknown
    assert gender["auc_gap"] is None and not gender["flagged"]  # only one group


def test_small_groups_get_no_auc() -> None:
    videos, labels = videos_and_labels(n=24)
    res = group_metrics(videos, labels, (0.3, 0.7), min_videos=20)
    assert all(g["auc"] is None for g in res["attributes"]["skin_tone"]["groups"].values())


def test_cli_writes_report_and_release_card_uses_it(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    Calibration(temperature=1.0, likely_real_below=0.3, likely_fake_above=0.7).save(
        run / "calibration.json"
    )
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "results.json").write_text(
        json.dumps({"checkpoint": str(run / "best.pt"), "video_score": "mean", "sets": {}})
    )
    videos, labels = videos_and_labels()
    frames = videos.rename(columns={"score": "prob"}).loc[videos.index.repeat(2)]
    frames.to_parquet(eval_dir / "celebdf_v2_frames.parquet")
    labels.to_csv(tmp_path / "labels.csv", index=False)

    fairness.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--labels",
            f"celebdf_v2={tmp_path / 'labels.csv'}",
            f"dfdc={tmp_path / 'labels.csv'}",
        ]
    )
    fair = json.loads((eval_dir / "fairness.json").read_text())
    assert fair["sets"]["celebdf_v2"]["attributes"]["skin_tone"]["flagged"]
    assert fair["sets"]["dfdc"]["status"] == "not evaluated"
    md = (eval_dir / "fairness.md").read_text()
    assert "**FLAGGED**" in md and "| A | 100 (50 / 50) |" in md

    template = Path("docs/MODEL_CARD.md").read_text(encoding="utf-8")
    card = fill_card(
        template,
        {"sets": {}},
        {"temperature": 1.0, "likely_real_below": 0.3, "likely_fake_above": 0.7},
        {},
        None,
        fair,
    )
    assert "**celebdf_v2, skin_tone**" in card
    assert "fairness.json" not in card.split("## Fairness")[1].split("##")[0].split("-->")[1]


def test_labels_need_video_id(tmp_path: Path) -> None:
    eval_dir = tmp_path
    (eval_dir / "results.json").write_text(
        json.dumps({"checkpoint": str(tmp_path / "x.pt"), "sets": {}})
    )
    pd.DataFrame({"video_id": ["a"], "label": [0], "prob": [0.1]}).to_parquet(
        eval_dir / "s_frames.parquet"
    )
    pd.DataFrame({"id": ["a"], "gender": ["x"]}).to_csv(tmp_path / "l.csv", index=False)
    with pytest.raises(ValueError, match="video_id"):
        fairness.fairness_for_eval(eval_dir, {"s": tmp_path / "l.csv"})


def test_markdown_handles_missing_set() -> None:
    md = fairness_markdown(
        {
            "cutoffs": [0.3, 0.7],
            "min_videos": 20,
            "gap_flag": 0.05,
            "sets": {"dfdc": {"status": "not evaluated"}},
        }
    )
    assert "**dfdc**: not evaluated." in md
