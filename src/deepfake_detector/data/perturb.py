"""Image degradations at rising strength for the robustness sweep (FR-E6).

Applied to the face crops, not the full frames, which is a common approximation: it degrades
exactly what the model sees, but compression of a full frame can differ slightly around the
crop edges.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Kind = Literal["jpeg", "blur", "downscale", "noise", "contrast"]


def perturb(img: np.ndarray, kind: Kind, strength: float, seed: int = 0) -> np.ndarray:
    """Return a degraded copy of an RGB uint8 image.

    jpeg: quality (lower is worse); blur: Gaussian sigma in pixels; downscale: scale factor
    before upscaling back; noise: Gaussian std as a fraction of 255; contrast: factor (lower is
    worse).
    """
    if kind == "jpeg":
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(strength)]
        )
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert ok and decoded is not None
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), float(strength))
    if kind == "downscale":
        h, w = img.shape[:2]
        small = cv2.resize(
            img,
            (max(1, int(w * strength)), max(1, int(h * strength))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        noisy = img.astype(np.float32) + rng.normal(0, strength * 255, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "contrast":
        f = img.astype(np.float32)
        mean = f.mean(axis=(0, 1), keepdims=True)
        return np.clip((f - mean) * strength + mean, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown perturbation: {kind}")


class PerturbTransform:
    """Dataset transform applying one degradation; labels are unchanged."""

    def __init__(self, kind: Kind, strength: float) -> None:
        self.kind, self.strength = kind, strength

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray | None, label: int
    ) -> tuple[np.ndarray, int]:
        return perturb(img, self.kind, self.strength), label
