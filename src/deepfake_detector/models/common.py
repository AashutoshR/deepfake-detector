"""Shared detector interface: every model takes 0 to 1 RGB tensors and returns logits."""

from __future__ import annotations

from typing import Any

import timm
import torch
from timm.data import resolve_data_config
from torch import nn

# Classes of the optional method head. "sbi" marks self-blended fakes made during training.
METHOD_CLASSES = ("real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "sbi")
IGNORE_INDEX = -100


def method_target(methods: list[str]) -> torch.Tensor:
    """Map method names to head targets; unknown methods are ignored by the loss."""
    lookup = {m: i for i, m in enumerate(METHOD_CLASSES)}
    return torch.tensor([lookup.get(m, IGNORE_INDEX) for m in methods], dtype=torch.long)


class Detector(nn.Module):
    """Backbone producing a feature vector, a real/fake head and an optional method head.

    Input normalization lives inside the model so data loaders and inference stay identical
    for every backbone.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        method_head: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        cfg = resolve_data_config({}, model=backbone)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, 1)
        self.method_head = nn.Linear(feature_dim, len(METHOD_CLASSES)) if method_head else None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.dropout(self.features(x))
        out = {"logit": self.head(f).squeeze(-1)}
        if self.method_head is not None:
            out["method_logits"] = self.method_head(f)
        return out


def create_backbone(name: str, pretrained: bool, **kwargs: Any) -> nn.Module:
    """timm model without its classifier (num_classes=0 returns pooled features)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
