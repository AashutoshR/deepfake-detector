import io
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from deepfake_detector.data.datasets import dataset_names, load_manifests
from deepfake_detector.data.indexers import INDEXERS, index_openforensics, index_wilddeepfake
from deepfake_detector.data.open_sets import crop_box, unpack_openforensics, unpack_wdf_tar
from deepfake_detector.open_model import DATASETS, OPEN_NOTE, model_markdown, train_overrides
from deepfake_detector.train import load_config


def _png(seed: int = 0, size: int = 16) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_unpack_wdf_tar_samples_frames_and_ignores_bad_paths(tmp_path: Path) -> None:
    tar = tmp_path / "101.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        members = {f"./101/fake/144/{i}.png": _png(i) for i in range(1000, 1020)}
        members["./101/fake/145/7.png"] = _png(1)
        members["./101/fake/145/notes.txt"] = b"x"
        members["../../evil/1/2/3.png"] = _png(2)
        members["./101/fake/146/8.png"] = b"not an image"
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = tmp_path / "wdf"
    assert unpack_wdf_tar(tar, out, "test", "fake", frames=4) == 5  # 4 from seq 144, 1 from 145
    seqs = sorted(p.name for p in (out / "test" / "fake").iterdir())
    assert seqs == ["101_144", "101_145"]
    assert len(list((out / "test" / "fake" / "101_144").glob("*.png"))) == 4
    assert not (tmp_path / "evil").exists()


def test_index_wilddeepfake_keeps_videos_in_one_split(tmp_path: Path) -> None:
    for split, kind in [("train", "real"), ("train", "fake"), ("test", "fake")]:
        for v in range(30):
            for s in range(2):
                d = tmp_path / split / kind / f"{v}_{s}"
                d.mkdir(parents=True)
                (d / "1.png").write_bytes(b"")
    recs = index_wilddeepfake(tmp_path)
    assert len(recs) == 180 and INDEXERS["wilddeepfake"] is index_wilddeepfake
    by_video: dict[tuple[str, ...], set[str]] = {}
    for r in recs:
        by_video.setdefault(r.persons, set()).add(r.split)
    assert all(len(s) == 1 for s in by_video.values())
    assert {r.split for r in recs} == {"train", "val", "test"}
    assert all(r.video_id.startswith("wdf/") for r in recs)
    assert {r.method for r in recs if r.label == 1} == {"wilddeepfake"}


def test_crop_box_grows_and_clips() -> None:
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop_box(img, [50, 20, 40, 40], 0.5).shape == (80, 80, 3)
    assert crop_box(img, [0, 0, 40, 40], 0.5).shape == (60, 60, 3)


def test_unpack_openforensics_one_file_per_labelled_face(tmp_path: Path) -> None:
    img = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    labels = {
        "categories": [{"id": 0, "name": "Real"}, {"id": 1, "name": "Fake"}],
        "images": [{"id": 7, "file_name": "Images/Val/abc123.jpg", "width": 300, "height": 200}],
        "annotations": [
            {"image_id": 7, "category_id": 1, "bbox": [20, 20, 60, 60]},
            {"image_id": 7, "category_id": 0, "bbox": [150, 50, 70, 80]},
            {"image_id": 7, "category_id": 1, "bbox": [250, 150, 10, 10]},  # too small
        ],
    }
    (tmp_path / "Val_poly.json").write_text(json.dumps(labels))
    zp = tmp_path / "Val.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("Images/Val/abc123.jpg", jpg.tobytes())
        zf.writestr("Images/Val/unlabelled.jpg", jpg.tobytes())
        zf.writestr("../evil.jpg", jpg.tobytes())
    out = tmp_path / "of"
    assert unpack_openforensics([zp], tmp_path / "Val_poly.json", out, "val") == 2
    assert [p.name for p in (out / "val" / "fake").iterdir()] == ["abc123_0.jpg"]
    assert [p.name for p in (out / "val" / "real").iterdir()] == ["abc123_1.jpg"]
    recs = index_openforensics(out)
    assert {r.persons for r in recs} == {("of:abc123",)}
    assert {(r.label, r.method) for r in recs} == {(1, "openforensics"), (0, "real")}


def test_load_manifests_stacks_datasets(tmp_path: Path) -> None:
    for name, n in [("a", 2), ("b", 3)]:
        (tmp_path / name).mkdir()
        pd.DataFrame(
            {"video_id": [f"{name}{i}" for i in range(n)], "split": "train", "method": "real",
             "compression": "na", "label": 0}
        ).to_parquet(tmp_path / name / "manifest.parquet")  # fmt: skip
    assert len(load_manifests(tmp_path, "a")) == 2
    assert len(load_manifests(tmp_path, ["a", "b"], split="train")) == 5
    assert dataset_names("a") == ["a"] and dataset_names(("a", "b")) == ["a", "b"]


def test_open_model_config_and_report() -> None:
    t = load_config(overrides=train_overrides(3, "o")).train
    assert list(t.dataset) == list(DATASETS) and t.sbi.enabled and not t.sbi.only
    assert t.leave_out_method is None and t.epochs == 3
    m = {"video_auc": 0.9, "video_eer": 0.1, "video_acc": 0.8, "video_ece": 0.05, "n_videos": 4.0}
    md = model_markdown(
        {"checkpoint": "c", "temperature": 1.0, "bands": [0.3, 0.7], "note": OPEN_NOTE,
         "test": {"wilddeepfake": {"all": m}}}
    )  # fmt: skip
    assert "| wilddeepfake | all | 0.9000 |" in md and chr(0x2014) not in md
