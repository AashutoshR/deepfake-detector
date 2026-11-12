import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from deepfake_detector import calibrate, evaluate
from deepfake_detector.calibrate import Calibration, calibrate_frames, fit_bands, fit_temperature
from deepfake_detector.data.perturb import PerturbTransform, perturb
from deepfake_detector.evaluate import bootstrap_auc_ci, results_markdown
from deepfake_detector.metrics import auc
from deepfake_detector.train import train
from tests.test_train import build_dataset, tiny_cfg


def test_fit_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 1.5, 20_000)
    labels = (rng.random(20_000) < 1 / (1 + np.exp(-true_logits))).astype(float)
    t = fit_temperature(true_logits * 3.0, labels)  # a model three times too confident
    assert t == pytest.approx(3.0, rel=0.1)


def test_calibration_round_trip(tmp_path: Path) -> None:
    cal = Calibration(temperature=2.0, likely_real_below=0.2, likely_fake_above=0.8)
    cal.save(tmp_path / "c.json")
    back = Calibration.load(tmp_path / "c.json")
    assert back == cal
    np.testing.assert_allclose(back.apply(np.array([0.0, 2.0])), [0.5, 1 / (1 + np.exp(-1))])


def test_fit_bands() -> None:
    y = np.array([0] * 100 + [1] * 100)
    # Overlapping classes: an uncertain band between the two cut-offs.
    p = np.concatenate([np.linspace(0.0, 0.6, 100), np.linspace(0.4, 1.0, 100)])
    lo, hi = fit_bands(p, y)
    assert lo < hi
    assert np.mean(p[y == 1] < lo) <= 0.05 and np.mean(p[y == 0] > hi) <= 0.05
    # Separable classes: the uncertain band keeps its minimum width around the middle.
    p = np.concatenate([np.linspace(0.0, 0.3, 100), np.linspace(0.7, 1.0, 100)])
    lo, hi = fit_bands(p, y)
    assert hi - lo == pytest.approx(0.2) and (lo + hi) / 2 == pytest.approx(0.5, abs=0.01)
    # Near an edge the band is shifted inward rather than cut off.
    p = np.concatenate([np.full(100, 0.0), np.full(100, 0.02)])
    lo, hi = fit_bands(p, y)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.2)


def test_calibrate_frames_reduces_ece_for_overconfident_model() -> None:
    rng = np.random.default_rng(1)
    vids = np.repeat(np.arange(400), 4)
    true = rng.normal(0, 1.0, 400)
    labels = (rng.random(400) < 1 / (1 + np.exp(-true))).astype(int)
    frames = pd.DataFrame(
        {
            "video_id": vids,
            "label": labels[vids],
            "logit": 5.0 * true[vids] + rng.normal(0, 0.1, 1600),
        }
    )
    cal = calibrate_frames(frames)
    assert cal.temperature > 2
    assert cal.val_video_ece_after < cal.val_video_ece_before


@pytest.mark.parametrize(
    ("kind", "strength"),
    [("jpeg", 30), ("blur", 2.0), ("downscale", 0.25), ("noise", 0.05), ("contrast", 0.4)],
)
def test_perturb_changes_image_keeps_shape(kind: str, strength: float) -> None:
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = perturb(img, kind, strength)  # type: ignore[arg-type]
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)
    img2, label = PerturbTransform(kind, strength)(img, None, 1)  # type: ignore[arg-type]
    assert label == 1 and img2.shape == img.shape


def test_perturb_unknown() -> None:
    with pytest.raises(ValueError):
        perturb(np.zeros((8, 8, 3), np.uint8), "rotate", 1)  # type: ignore[arg-type]


def test_bootstrap_ci_contains_auc() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    lo, hi = bootstrap_auc_ci(y, s, n=200)
    assert lo < auc(y, s) < hi
    assert np.isnan(bootstrap_auc_ci(np.ones(5), np.ones(5))[0])


def test_results_markdown_marks_missing_sets() -> None:
    md = results_markdown(
        {
            "checkpoint": "x.pt",
            "temperature": 1.0,
            "leave_out_method": None,
            "video_score": "mean",
            "sets": {"dfdc": {"status": "not available"}},
        }
    )
    assert "| dfdc | not available |" in md
    assert "not proof" in md


def test_calibrate_and_evaluate_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "data"
    build_dataset(root)
    run = Path(train(tiny_cfg(root, "train.epochs=1"), pretrained_weights=False)["run_dir"])
    ckpt = run / "best.pt"

    calibrate.main(["--checkpoint", str(ckpt)])
    cal = Calibration.load(run / "calibration.json")
    assert cal.temperature > 0 and cal.likely_real_below <= cal.likely_fake_above

    eval_cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())
    eval_cfg.update(num_workers=0, batch_size=8)
    eval_cfg["bootstrap"]["n"] = 50
    eval_cfg["robustness"]["sweeps"] = {"jpeg": [50], "blur": [1.0]}
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(eval_cfg))
    out = tmp_path / "eval_out"
    evaluate.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(cfg_path),
            "--out",
            str(out),
            "--sets",
            "ffpp_c23",
            "celebdf_v2",
            "--robustness",
        ]
    )

    results = json.loads((out / "results.json").read_text())
    assert results["temperature"] == pytest.approx(cal.temperature)
    ff = results["sets"]["ffpp_c23"]
    assert ff["status"] == "ok" and ff["n_videos"] == 4
    assert set(ff["per_method_auc"]) == {"Deepfakes", "FaceSwap"}
    assert results["sets"]["celebdf_v2"]["status"] == "not available"
    assert set(results["robustness"]["sweeps"]) == {"jpeg", "blur"}
    for f in [
        "results.md",
        "roc.png",
        "calibration_ffpp_c23.png",
        "ffpp_c23_frames.parquet",
        "robustness.png",
        "robustness.json",
    ]:
        assert (out / f).exists(), f
