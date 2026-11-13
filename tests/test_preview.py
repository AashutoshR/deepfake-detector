import io
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from api.server import create_app
from fastapi.testclient import TestClient

from deepfake_detector.data.datasets import FaceCropDataset, add_sbi_copies
from deepfake_detector.data.faces import FaceDetection
from deepfake_detector.data.indexers import INDEXERS, index_ffhq
from deepfake_detector.predict import Predictor
from deepfake_detector.preview import (
    PREVIEW_NOTE,
    preview_markdown,
    shrink,
    train_overrides,
    unpack_shard,
)
from deepfake_detector.train import load_config


class SpreadDetector:
    """One face in the middle with landmarks on an ellipse, so a blend mask has an area."""

    def detect(self, rgb: np.ndarray) -> list[FaceDetection]:
        h, w = rgb.shape[:2]
        t = np.linspace(0, 2 * np.pi, 68, endpoint=False)
        lm = np.stack([w / 2 + w * 0.15 * np.cos(t), h / 2 + h * 0.2 * np.sin(t)], 1)
        box = (w * 0.3, h * 0.25, w * 0.7, h * 0.75)
        return [FaceDetection(box=box, score=0.99, landmarks=lm.astype(np.float32))]


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_shrink_keeps_aspect_and_never_enlarges() -> None:
    assert shrink(np.zeros((1024, 512, 3), np.uint8), 512).shape == (512, 256, 3)
    small = np.zeros((100, 80, 3), np.uint8)
    assert shrink(small, 512) is small


def test_unpack_shard_keeps_only_images(tmp_path: Path) -> None:
    tar_path = tmp_path / "00000.tar"
    ok, png = cv2.imencode(".png", _noise(64, 64))
    assert ok
    with tarfile.open(tar_path, "w") as tf:
        for name, data in [
            ("00001.png", png.tobytes()),
            ("00001.json", b"{}"),
            ("../evil.png", png.tobytes()),
            ("00002.webp", b"not an image"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = tmp_path / "real"
    assert unpack_shard(tar_path, out, max_side=32) == 2  # 00001 and the sanitised evil
    assert sorted(p.name for p in out.iterdir()) == ["00001.jpg", "evil.jpg"]
    assert cv2.imread(str(out / "00001.jpg")).shape == (32, 32, 3)
    assert not (tmp_path / "evil.png").exists()
    assert unpack_shard(tar_path, out) == 2  # already there: skipped, still counted


def test_index_ffhq_splits_by_id(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    for i in range(40):
        (real / f"{i:05d}.jpg").write_bytes(b"")
    recs = index_ffhq(tmp_path)
    splits = pd.Series([r.split for r in recs]).value_counts().to_dict()
    assert splits == {"train": 36, "val": 2, "test": 2}
    assert all(r.label == 0 and r.compression == "na" for r in recs)
    assert len({r.persons for r in recs}) == 40
    assert INDEXERS["ffhq"] is index_ffhq


def _crop_table(tmp_path: Path, n: int = 3) -> pd.DataFrame:
    rows = []
    t = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    lm = np.stack([32 + 14 * np.cos(t), 32 + 18 * np.sin(t)], 1).reshape(-1).tolist()
    for i in range(n):
        path = tmp_path / f"c{i}.png"
        cv2.imwrite(str(path), _noise(64, 64, i))
        rows.append(
            {"video_id": f"real/{i}", "label": 0, "method": "real", "crop_path": path.name,
             "landmarks": lm}
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_sbi_copies_are_fixed_fakes(tmp_path: Path) -> None:
    df = add_sbi_copies(_crop_table(tmp_path), seed=3)
    assert len(df) == 6
    fakes = df[df["label"] == 1]
    assert set(fakes["method"]) == {"sbi"} and fakes["video_id"].str.endswith("#sbi").all()
    assert (df.loc[df["label"] == 0, "sbi_seed"] == -1).all()

    ds = FaceCropDataset(df, tmp_path)
    real, fake = ds[0], ds[3]
    assert fake["label"].item() == 1 and fake["method"] == "sbi"
    assert not np.allclose(real["image"].numpy(), fake["image"].numpy())
    assert np.allclose(fake["image"].numpy(), FaceCropDataset(df, tmp_path)[3]["image"].numpy())


def test_train_overrides_make_a_real_only_sbi_run() -> None:
    cfg = load_config(overrides=train_overrides(3, "p"))
    t = cfg.train
    assert t.dataset == "ffhq" and list(t.compressions) == ["na"]
    assert t.sbi.only and t.val.sbi_fakes and t.epochs == 3 and t.run_name == "p"


def test_preview_markdown() -> None:
    md = preview_markdown(
        {
            "checkpoint": "runs/p/best.pt",
            "temperature": 1.5,
            "bands": [0.2, 0.8],
            "test_photos": 10,
            "test_metrics_self_blended": {"video_auc": 0.95, "n_videos": 20.0},
            "note": PREVIEW_NOTE,
        }
    )
    assert "| video_auc | 0.9500 |" in md and PREVIEW_NOTE in md and chr(0x2014) not in md


# ----------------------------------------------------------------------------- test fakes


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "face.png"
    cv2.imwrite(str(p), _noise(120, 100))
    return p


def test_make_test_fake(checkpoint: Path, photo: Path) -> None:
    pred = Predictor(checkpoint, device="cpu", detector=SpreadDetector())
    pair = pred.make_test_fake(photo, seed=1)
    assert pair is not None
    original, blended = pair
    assert original.shape == blended.shape == (120, 100, 3)
    assert not np.array_equal(original, blended)
    assert pred.model_note is None


def test_api_test_fake_and_preview_note(checkpoint: Path, photo: Path) -> None:
    pred = Predictor(checkpoint, device="cpu", detector=SpreadDetector())
    pred.model_note = PREVIEW_NOTE
    client = TestClient(create_app(pred))
    assert client.get("/health").json()["model_note"] == PREVIEW_NOTE
    r = client.post("/test-fake", files={"file": ("face.png", photo.read_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["original_png_base64"] and body["blended_png_base64"]
    r = client.post("/test-fake", files={"file": ("clip.mp4", b"x", "video/mp4")})
    assert r.status_code == 415

    class NoFace:
        def detect(self, rgb: np.ndarray) -> list[FaceDetection]:
            return []

    pred.detector = NoFace()
    r = client.post("/test-fake", files={"file": ("face.png", photo.read_bytes(), "image/png")})
    assert r.status_code == 422


def test_demo_finds_newest_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import demo

    monkeypatch.chdir(tmp_path)
    assert demo.newest_checkpoint() is None
    for i, name in enumerate(["a", "b"]):
        (tmp_path / "runs" / name).mkdir(parents=True)
        p = tmp_path / "runs" / name / "best.pt"
        p.write_bytes(b"")
        import os

        os.utime(p, (1_000 + i, 1_000 + i))
    assert demo.newest_checkpoint() == Path("runs/b/best.pt")
