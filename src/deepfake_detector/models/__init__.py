"""Models: ViT-B/16 detector and the Xception and EfficientNet-B4 baselines."""

from __future__ import annotations

from typing import Any

from deepfake_detector.models.common import Detector


def build_model(cfg: dict[str, Any], pretrained_weights: bool = True) -> Detector:
    """Build any detector from the model config section."""
    arch = cfg.get("arch", "vit")
    if arch == "vit":
        from deepfake_detector.models.vit import build_vit

        return build_vit(cfg, pretrained_weights)
    if arch in ("xception", "efficientnet_b4"):
        from deepfake_detector.models.baselines import build_baseline

        return build_baseline(cfg, pretrained_weights)
    raise ValueError(f"unknown arch: {arch}")
