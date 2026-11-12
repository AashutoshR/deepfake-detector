"""Self-blended images: blend a real face with a changed copy of itself (FR-T4).

Follows the idea of Shiohara and Yamasaki (CVPR 2022): take a real face, make a slightly altered
copy of it (colour shift, sharpen or blur, small resize and shift), and paste the copy back over
the face through a soft landmark mask. The result carries blending marks (seams, colour and
frequency mismatch) without any one generator's style, which helps cross-dataset generalization.
"""

from __future__ import annotations

import cv2
import numpy as np

BLEND_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)


def hull_mask(shape: tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
    """Filled convex hull of the landmarks as a float mask in [0, 1] of size (H, W)."""
    mask = np.zeros(shape, np.float32)
    pts = np.round(landmarks).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


def deform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly grow or shrink the mask and soften its edge."""
    h, w = mask.shape
    k = max(1, int(round(rng.uniform(0.01, 0.05) * min(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.erode(mask, kernel) if rng.random() < 0.5 else cv2.dilate(mask, kernel)
    blur = int(rng.integers(2, max(3, min(h, w) // 8))) * 2 + 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    peak = mask.max()
    return mask / peak if peak > 0 else mask


def alter_source(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make the slightly changed copy: colour shift, then sharpen or blur, then resize and shift."""
    h, w = img.shape[:2]
    out: np.ndarray = img.astype(np.float32)
    out += rng.uniform(-20, 20, size=3).astype(np.float32)  # per-channel colour shift
    out = (out - out.mean()) * rng.uniform(0.9, 1.1) + out.mean()  # contrast
    out = np.clip(out, 0, 255)

    if rng.random() < 0.5:
        blurred = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5))
        out = np.clip(out + (out - blurred) * rng.uniform(0.3, 1.0), 0, 255)  # sharpen
    else:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))

    scale = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-0.03, 0.03, size=2) * np.array([w, h])
    affine = np.array(
        [[scale, 0, (1 - scale) * w / 2 + tx], [0, scale, (1 - scale) * h / 2 + ty]], np.float32
    )
    out = cv2.warpAffine(out, affine, (w, h), borderMode=cv2.BORDER_REFLECT)
    return out


def self_blend(
    img: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (blended uint8 image, blend mask). The blended image is a synthetic fake."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    mask = deform_mask(hull_mask((h, w), landmarks), rng)
    mask *= rng.choice(BLEND_RATIOS)
    source = alter_source(img, rng)
    m3 = mask[..., None]
    blended = source * m3 + img.astype(np.float32) * (1 - m3)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8), mask
