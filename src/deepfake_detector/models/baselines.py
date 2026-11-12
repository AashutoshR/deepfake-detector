"""Xception and EfficientNet-B4 baselines on the same interface (FR-M4).

Both are fully fine-tuned from ImageNet weights on the same 224 x 224 crops and pipeline as the
ViT, so differences come from the model, not the data.
"""

from __future__ import annotations

import contextlib
from typing import Any

from deepfake_detector.models.common import Detector, create_backbone

BASELINES = {
    "xception": "legacy_xception.tf_in1k",
    "efficientnet_b4": "tf_efficientnet_b4.ns_jft_in1k",
}


def build_baseline(cfg: dict[str, Any], pretrained_weights: bool = True) -> Detector:
    name = cfg.get("timm_name") or BASELINES[cfg["arch"]]
    backbone = create_backbone(name, pretrained=pretrained_weights)
    if cfg.get("grad_checkpointing", False):
        # Some timm CNNs do not support it; they fit in memory without it.
        with contextlib.suppress(AssertionError, NotImplementedError):
            backbone.set_grad_checkpointing(True)  # type: ignore[operator]
    return Detector(
        backbone,
        feature_dim=backbone.num_features,  # type: ignore[arg-type]
        method_head=cfg.get("method_head", {}).get("enabled", False),
        dropout=cfg.get("dropout", 0.0),
    )
