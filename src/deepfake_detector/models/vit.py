"""ViT-B/16 from timm with LoRA via peft, CLS or mean-patch pooling (FR-M1 to FR-M3, FR-M5)."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from deepfake_detector.models.common import Detector, create_backbone

PRETRAINED = {
    "imagenet21k": "vit_base_patch16_224.augreg_in21k",
    "clip": "vit_base_patch16_clip_224.openai",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "none": "vit_base_patch16_224",
}
POOLING = {"cls": "token", "mean_patch": "avg"}


def _set_trainable(backbone: nn.Module, mode: str) -> None:
    if mode == "full":
        for p in backbone.parameters():
            p.requires_grad = True
        return
    for name, p in backbone.named_parameters():
        if mode == "lora":
            p.requires_grad = "lora_" in name
        elif mode == "last_layer":
            n_blocks = len(backbone.blocks)  # type: ignore[arg-type]
            p.requires_grad = name.startswith((f"blocks.{n_blocks - 1}.", "norm.", "fc_norm."))
        else:
            raise ValueError(f"unknown finetune mode: {mode}")


def build_vit(cfg: dict[str, Any], pretrained_weights: bool = True) -> Detector:
    """Build the ViT detector from the model config (see configs/model.yaml).

    pretrained_weights=False skips the weight download (tests and memory benchmarks).
    """
    tag = cfg.get("pretrained", "imagenet21k")
    name = cfg.get("timm_name") or PRETRAINED[tag]
    use_weights = pretrained_weights and tag != "none"
    backbone = create_backbone(
        name,
        pretrained=use_weights,
        global_pool=POOLING[cfg.get("pooling", "cls")],
        img_size=cfg.get("img_size", 224),
    )
    if cfg.get("grad_checkpointing", False):
        backbone.set_grad_checkpointing(True)  # type: ignore[operator]

    ft = cfg.get("finetune", {})
    mode = ft.get("mode", "lora")
    if mode == "lora":
        lora = ft.get("lora", {})
        config = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.1),
            target_modules=list(lora.get("target_modules", ["qkv"])),
        )
        backbone = inject_adapter_in_model(config, backbone)
    _set_trainable(backbone, mode)

    return Detector(
        backbone,
        feature_dim=backbone.num_features,  # type: ignore[arg-type]
        method_head=cfg.get("method_head", {}).get("enabled", False),
        dropout=cfg.get("dropout", 0.0),
    )
