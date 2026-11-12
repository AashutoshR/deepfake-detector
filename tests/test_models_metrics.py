import numpy as np
import pandas as pd
import pytest
import torch

from deepfake_detector.metrics import (
    auc,
    auc_per_method,
    ece,
    eer,
    log_loss,
    summarize,
    video_scores,
)
from deepfake_detector.models import build_model
from deepfake_detector.models.common import IGNORE_INDEX, count_parameters, method_target

TINY = {"arch": "vit", "pretrained": "none", "timm_name": "vit_tiny_patch16_224", "img_size": 32}


@pytest.mark.parametrize("mode", ["lora", "full", "last_layer"])
def test_vit_finetune_modes(mode: str) -> None:
    model = build_model({**TINY, "finetune": {"mode": mode}}, pretrained_weights=False)
    trainable, total = count_parameters(model)
    if mode == "full":
        assert trainable == total
    else:
        assert 0 < trainable < total / 4
    if mode == "lora":
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("lora_" in n or n.startswith("head") for n in names)


@pytest.mark.parametrize("pooling", ["cls", "mean_patch"])
def test_vit_forward_backward_with_checkpointing(pooling: str) -> None:
    cfg = {**TINY, "pooling": pooling, "grad_checkpointing": True, "method_head": {"enabled": True}}
    model = build_model(cfg, pretrained_weights=False)
    out = model(torch.rand(2, 3, 32, 32))
    assert out["logit"].shape == (2,)
    assert out["method_logits"].shape == (2, 6)
    out["logit"].sum().backward()
    assert model.head.weight.grad is not None


@pytest.mark.parametrize("arch", ["xception", "efficientnet_b4"])
def test_baselines_share_interface(arch: str) -> None:
    model = build_model({"arch": arch}, pretrained_weights=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert out["logit"].shape == (1,)
    trainable, total = count_parameters(model)
    assert trainable == total


def test_unknown_arch_and_mode() -> None:
    with pytest.raises(ValueError):
        build_model({"arch": "resnet"})
    with pytest.raises(ValueError):
        build_model({**TINY, "finetune": {"mode": "half"}}, pretrained_weights=False)


def test_method_target() -> None:
    t = method_target(["real", "sbi", "Deepfakes", "unknown"])
    assert t.tolist() == [0, 5, 1, IGNORE_INDEX]


def test_auc_eer_basic() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert eer(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.3])))
    assert np.isnan(eer(np.array([0, 0]), np.array([0.2, 0.3])))


def test_ece_and_log_loss() -> None:
    y = np.array([0, 1, 0, 1])
    assert ece(y, np.array([0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.0)
    assert ece(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert np.isnan(ece(np.array([]), np.array([])))
    assert log_loss(y, np.array([0.5] * 4)) == pytest.approx(np.log(2))


def frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["r1", "r1", "r2", "r2", "f1", "f1", "f2", "f2"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "method": ["real"] * 4 + ["Deepfakes"] * 2 + ["FaceSwap"] * 2,
            "prob": [0.1, 0.3, 0.2, 0.9, 0.8, 0.6, 0.4, 0.95],
        }
    )


@pytest.mark.parametrize(("how", "f2"), [("mean", 0.675), ("median", 0.675), ("topk", 0.95)])
def test_video_scores(how: str, f2: float) -> None:
    v = video_scores(frames(), how=how, k=1).set_index("video_id")  # type: ignore[arg-type]
    assert v.loc["f2", "score"] == pytest.approx(f2)
    assert v.loc["f2", "label"] == 1
    with pytest.raises(ValueError):
        video_scores(frames(), how="max")  # type: ignore[arg-type]


def test_summarize_and_per_method() -> None:
    s = summarize(frames())
    assert s["n_videos"] == 4 and s["n_frames"] == 8
    assert 0 <= s["video_auc"] <= 1 and 0 <= s["video_ece"] <= 1
    per = auc_per_method(frames())
    assert set(per) == {"Deepfakes", "FaceSwap"}
