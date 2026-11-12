from pathlib import Path

import pandas as pd
import torch

from deepfake_detector.data.datasets import FaceCropDataset, balanced_sampler, load_manifest
from deepfake_detector.data.indexers import FAKE, REAL, VideoRecord
from deepfake_detector.data.preprocess import finalize, pending, process_video, shard_name
from tests.conftest import FixedDetector, write_video


def rec(path: Path, vid: str, label: int = REAL, split: str = "train") -> VideoRecord:
    method = "real" if label == REAL else "Deepfakes"
    return VideoRecord(path, vid, label, method, "ffpp", split, "c23", (f"ffpp:{vid}",))


def save(out: Path, r: VideoRecord, crops: pd.DataFrame, skips: pd.DataFrame) -> None:
    sub = "skips" if not skips.empty else "shards"
    d = out / "manifests" / "ffpp" / sub
    d.mkdir(parents=True, exist_ok=True)
    (skips if not skips.empty else crops).to_parquet(d / shard_name(r.video_id), index=False)


def test_process_video_writes_crops(tmp_path: Path, video: Path) -> None:
    r = rec(video, "real/000")
    crops, skips = process_video(r, FixedDetector(), tmp_path, frames=8, size=64)
    assert skips.empty
    assert len(crops) == 8
    first = tmp_path / crops.iloc[0]["crop_path"]
    assert first.exists()
    assert len(crops.iloc[0]["landmarks"]) == 136


def test_process_video_logs_no_face(tmp_path: Path, video: Path) -> None:
    crops, skips = process_video(rec(video, "real/001"), FixedDetector(empty=True), tmp_path, 4)
    assert crops.empty
    assert skips.iloc[0]["reason"] == "no_face"
    assert skips.iloc[0]["frames_read"] == 4


def test_process_video_logs_read_error(tmp_path: Path) -> None:
    _, skips = process_video(rec(tmp_path / "nope.mp4", "real/002"), FixedDetector(), tmp_path)
    assert skips.iloc[0]["reason"].startswith("read_error")


def test_resume_and_finalize_and_dataset(tmp_path: Path) -> None:
    out = tmp_path / "data"
    v1, v2, v3 = (tmp_path / f"{i}.mp4" for i in range(3))
    for v in (v1, v2, v3):
        write_video(v, 10)
    recs = [rec(v1, "real/000"), rec(v2, "Deepfakes/000_001", FAKE), rec(v3, "real/009")]

    for r, detector in [(recs[0], FixedDetector()), (recs[1], FixedDetector())]:
        save(out, r, *process_video(r, detector, out, frames=4, size=32))
    assert pending(recs, out) == [recs[2]]

    save(out, recs[2], *process_video(recs[2], FixedDetector(empty=True), out, frames=4))
    assert pending(recs, out) == []

    crops, skips = finalize("ffpp", out)
    assert len(crops) == 8 and len(skips) == 1
    assert (out / "manifests/ffpp/manifest.parquet").exists()

    df = load_manifest(
        out / "manifests/ffpp/manifest.parquet", split="train", leave_out_method="Deepfakes"
    )
    assert set(df["method"]) == {"real"}

    full = load_manifest(out / "manifests/ffpp/manifest.parquet")
    ds = FaceCropDataset(full, out)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image"].dtype == torch.float32 and item["image"].max() <= 1.0
    sampler = balanced_sampler(full)
    assert len(list(sampler)) == len(full)
