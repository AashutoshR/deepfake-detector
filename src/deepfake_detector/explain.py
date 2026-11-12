"""Attention rollout heatmaps for the ViT detector (FR-X1).

Rollout (Abnar and Zuidema, 2020) multiplies the head-averaged attention of every layer, with
the residual connection added, to estimate how much each image patch feeds the final score.
It shows where the model looks, not why: a heatmap is a hint for a human, not evidence.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch

from deepfake_detector.models.common import Detector


def attention_rollout(model: Detector, x: torch.Tensor) -> np.ndarray:
    """Return heatmaps (B, gh, gw) in [0, 1] for 0 to 1 RGB input x (B, 3, H, W)."""
    backbone: Any = model.backbone
    # CNNs such as EfficientNet also have .blocks, so check for the patch embedding instead.
    if not hasattr(backbone, "patch_embed"):
        raise ValueError("attention rollout needs a ViT backbone")
    blocks = backbone.blocks

    attns: list[torch.Tensor] = []

    def make_hook(attn: Any) -> Any:
        def hook(_: Any, __: Any, out: torch.Tensor) -> None:
            b, n, _c = out.shape
            qkv = out.reshape(b, n, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
            a = (q.float() @ k.float().transpose(-2, -1)) * attn.scale
            attns.append(a.softmax(-1).mean(1))

        return hook

    handles = [blk.attn.qkv.register_forward_hook(make_hook(blk.attn)) for blk in blocks]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    n = attns[0].shape[-1]
    eye = torch.eye(n, device=attns[0].device)
    rollout = eye.expand_as(attns[0]).clone()
    for a in attns:
        a = a + eye
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout

    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    if getattr(backbone, "global_pool", "token") == "avg":
        scores = rollout[:, prefix:, prefix:].mean(1)  # mean-pooled model: average patch rows
    else:
        scores = rollout[:, 0, prefix:]  # CLS token row
    g = int(math.isqrt(scores.shape[-1]))
    heat = scores[:, : g * g].reshape(-1, g, g)
    lo = heat.amin(dim=(1, 2), keepdim=True)
    hi = heat.amax(dim=(1, 2), keepdim=True)
    return ((heat - lo) / (hi - lo + 1e-8)).cpu().numpy()


def overlay(crop: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (any size, 0 to 1) over an RGB uint8 crop."""
    h, w = crop.shape[:2]
    up = cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(up, 0, 1) * 255), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(crop * (1 - alpha) + colored * alpha)
