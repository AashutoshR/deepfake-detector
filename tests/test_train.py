"""End-to-end training on a tiny synthetic dataset with a tiny ViT on CPU."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deepfake_detector.data.indexers import FAKE, REAL, VideoRecord
from deepfake_detector.data.manifest import merge_shards
from deepfake_detector.data.preprocess import process_video, shard_name
from deepfake_detector.train import cosine_with_warmup, epoch_indices, load_config, train
from tests.conftest import FixedDetector, write_video


def build_dataset(root: Path, size: int = 32) -> None:
    shards = root / "manifests" / "ffpp" / "shards"
    shards.mkdir(parents=True)
    n = 0
    for split in ("train", "val"):
        for label, method in [
            (REAL, "real"),
            (FAKE, "Deepfakes"),
            (REAL, "real"),
            (FAKE, "FaceSwap"),
        ]:
            n += 1
            path = root / "raw" / f"{n}.mp4"
            write_video(path, 6)
            vid = f"{method}/{n:03d}"
            rec = VideoRecord(path, vid, label, method, "ffpp", split, "c23", (f"ffpp:{n}",))
            crops, _ = process_video(rec, FixedDetector(), root, frames=4, size=size)
            crops.to_parquet(shards / shard_name(vid), index=False)
    merge_shards(shards, root / "manifests" / "ffpp" / "manifest.parquet")


def tiny_cfg(root: Path, *extra: str):
    return load_config(
        overrides=[
            f"data.root={root.as_posix()}",
            "model.pretrained=none",
            "model.timm_name=vit_tiny_patch16_224",
            "model.img_size=32",
            "train.epochs=2",
            "train.batch_size=4",
            "train.grad_accum=2",
            "train.num_workers=0",
            "train.precision=fp32",
            "train.samples_per_epoch=16",
            "train.val.frames_per_video=2",
            "train.augment.preset=basic",
            "train.log_every=1",
            f"train.out_dir={(root / 'runs').as_posix()}",
            "train.run_name=test",
            *extra,
        ]
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    build_dataset(root)
    return root


def test_train_end_to_end(data_root: Path) -> None:
    result = train(tiny_cfg(data_root, "model.method_head.enabled=true"), pretrained_weights=False)
    run = Path(result["run_dir"])
    for f in ["config.yaml", "env.json", "last.pt", "best.pt", "metrics.jsonl"]:
        assert (run / f).exists(), f
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[-1])
    assert rec["n_videos"] == 4 and set(rec["per_method"]) == {"Deepfakes", "FaceSwap"}
    env = json.loads((run / "env.json").read_text())
    assert "torch" in env and "git_commit" in env
    ckpt = torch.load(run / "last.pt", weights_only=True)
    assert ckpt["epoch"] == 2 and ckpt["batch_in_epoch"] == 0


def test_leave_one_method_out_and_resume(data_root: Path) -> None:
    cfg = tiny_cfg(data_root, "train.epochs=1", "train.leave_out_method=FaceSwap")
    result = train(cfg, pretrained_weights=False)
    run = Path(result["run_dir"])
    rec = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert set(rec["per_method"]) == {"Deepfakes"}  # held-out method not in validation

    # Extend the finished run by one epoch and resume from its checkpoint.
    ckpt = torch.load(run / "last.pt", weights_only=True)
    ckpt["config"]["train"]["epochs"] = 2
    torch.save(ckpt, run / "last.pt")
    train(cfg, resume=run / "last.pt")
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [1, 2]


def test_sbi_only_trains_on_real_crops(data_root: Path) -> None:
    result = train(
        tiny_cfg(data_root, "train.epochs=1", "train.sbi.only=true"), pretrained_weights=False
    )
    assert Path(result["run_dir"], "best.pt").exists()


def test_epoch_indices_balanced_and_reproducible() -> None:
    labels = np.array([0] * 10 + [1] * 40)
    a = epoch_indices(labels, 4000, seed=0, epoch=3)
    assert a == epoch_indices(labels, 4000, seed=0, epoch=3)
    assert a != epoch_indices(labels, 4000, seed=0, epoch=4)
    share_real = np.mean(labels[a] == 0)
    assert 0.45 < share_real < 0.55


def test_cosine_with_warmup() -> None:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=2, total_steps=10)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert lrs[-1] < 0.05


def test_config_overrides_resolve(tmp_path: Path) -> None:
    cfg = load_config(overrides=[f"data.root={tmp_path.as_posix()}", "train.epochs=3"])
    assert cfg.train.epochs == 3
    assert cfg.data.manifest_dir == f"{tmp_path.as_posix()}/manifests"
    assert isinstance(pd.__version__, str)
