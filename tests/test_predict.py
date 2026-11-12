import base64
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from api.server import create_app
from fastapi.testclient import TestClient

from deepfake_detector.calibrate import Calibration
from deepfake_detector.explain import attention_rollout, overlay
from deepfake_detector.models import build_model
from deepfake_detector.predict import (
    NOT_PROOF_NOTE,
    Band,
    BandCutoffs,
    Prediction,
    Predictor,
    build_parser,
    main,
    to_band,
)
from deepfake_detector.train import train
from tests.conftest import FixedDetector, write_video
from tests.test_train import build_dataset, tiny_cfg


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, Band.LIKELY_REAL),
        (0.29, Band.LIKELY_REAL),
        (0.3, Band.UNCERTAIN),
        (0.7, Band.UNCERTAIN),
        (0.71, Band.LIKELY_MANIPULATED),
        (1.0, Band.LIKELY_MANIPULATED),
    ],
)
def test_to_band_default_cutoffs(p: float, expected: Band) -> None:
    assert to_band(p) is expected


def test_to_band_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        to_band(1.5)


def test_cutoffs_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        BandCutoffs(likely_real_below=0.8, likely_fake_above=0.2)


def test_prediction_always_carries_note() -> None:
    d = Prediction(probability=0.9, band=Band.LIKELY_MANIPULATED).to_dict()
    assert d["note"] == NOT_PROOF_NOTE
    assert d["band"] == "likely manipulated"
    assert Prediction(None, None, status="no_face").to_dict()["band"] is None


def test_parser_requires_input_and_checkpoint() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# ----------------------------------------------------------------------------- heatmaps

TINY = {"arch": "vit", "pretrained": "none", "timm_name": "vit_tiny_patch16_224", "img_size": 64}


@pytest.mark.parametrize("pooling", ["cls", "mean_patch"])
def test_attention_rollout_shape_and_range(pooling: str) -> None:
    model = build_model({**TINY, "pooling": pooling}, pretrained_weights=False)
    heat = attention_rollout(model, torch.rand(2, 3, 64, 64))
    assert heat.shape == (2, 4, 4)
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
    assert model.training  # restores the previous mode


def test_attention_rollout_needs_vit() -> None:
    model = build_model({"arch": "efficientnet_b4"}, pretrained_weights=False)
    with pytest.raises(ValueError):
        attention_rollout(model, torch.rand(1, 3, 64, 64))


def test_overlay() -> None:
    out = overlay(np.zeros((64, 64, 3), np.uint8), np.random.rand(4, 4))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8 and out.max() > 0


# ----------------------------------------------------------------------------- predictor, CLI, API


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pred") / "data"
    build_dataset(root)
    run = Path(train(tiny_cfg(root, "train.epochs=1"), pretrained_weights=False)["run_dir"])
    Calibration(temperature=2.0, likely_real_below=0.4, likely_fake_above=0.6).save(
        run / "calibration.json"
    )
    return run / "best.pt"


@pytest.fixture(scope="module")
def predictor(checkpoint: Path) -> Predictor:
    return Predictor(checkpoint, device="cpu", detector=FixedDetector(), frames=6)


def test_predict_video(predictor: Predictor, tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    write_video(video, 20)
    r = predictor.predict(video)
    assert r.status == "ok" and r.faces_found == 6 and r.frames_read == 6
    assert r.probability is not None and 0 <= r.probability <= 1
    assert r.band is to_band(r.probability, BandCutoffs(0.4, 0.6))
    assert len(r.frame_scores) == 6 and r.frames == sorted(r.frames)
    assert r.cutoffs == {"likely_real_below": 0.4, "likely_fake_above": 0.6}
    png = base64.b64decode(r.heatmap_png_base64 or "")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (32, 32, 3)
    assert predictor.calibrated


def test_predict_image_and_no_face(checkpoint: Path, tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    cv2.imwrite(str(img), np.full((120, 160, 3), 90, np.uint8))
    r = Predictor(checkpoint, device="cpu", detector=FixedDetector()).predict(img, heatmap=False)
    assert r.faces_found == 1 and r.heatmap_png_base64 is None
    none = Predictor(checkpoint, device="cpu", detector=FixedDetector(empty=True)).predict(img)
    assert none.status == "no_face" and none.probability is None and none.band is None


def test_cli_prints_json_and_saves_heatmap(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("deepfake_detector.data.faces.SFDDetector", lambda device: FixedDetector())
    video = tmp_path / "v.mp4"
    write_video(video, 10)
    heat = tmp_path / "heat.png"
    main(
        [
            "--input",
            str(video),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--frames",
            "4",
            "--heatmap",
            str(heat),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["faces_found"] == 4 and out["note"] == NOT_PROOF_NOTE
    assert "heatmap_png_base64" not in out and heat.exists()


def _dfd_temp_files() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("dfd_*")}


def test_api_predict_deletes_upload(predictor: Predictor, tmp_path: Path) -> None:
    client = TestClient(create_app(predictor))
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["model_loaded"] and health["note"] == NOT_PROOF_NOTE

    video = tmp_path / "v.mp4"
    write_video(video, 12)
    before = _dfd_temp_files()
    r = client.post("/predict", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["note"] == NOT_PROOF_NOTE
    assert body["band"] in {b.value for b in Band}
    assert _dfd_temp_files() == before  # FR-I5: upload removed


def test_api_rejects_bad_type_and_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DFD_CHECKPOINT", raising=False)
    client = TestClient(create_app())
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/predict", files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
