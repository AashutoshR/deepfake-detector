"""Augmentation presets and the training transform (FR-T3, FR-T4).

basic: horizontal flip only.
full:  flip plus JPEG recompression, blur, downscale then upscale, noise, colour jitter, cutout.
Written for albumentations 2.x.
"""

from __future__ import annotations

from typing import Literal

import albumentations as A
import numpy as np

from deepfake_detector.data.indexers import FAKE, REAL
from deepfake_detector.data.sbi import self_blend

Preset = Literal["basic", "full"]


def build_augment(preset: Preset = "full") -> A.Compose:
    if preset == "basic":
        return A.Compose([A.HorizontalFlip(p=0.5)])
    if preset != "full":
        raise ValueError(f"unknown augmentation preset: {preset}")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_range=(30, 100), p=0.5),
            A.OneOf(
                [A.GaussianBlur(blur_limit=(3, 7)), A.Downscale(scale_range=(0.25, 0.75))],
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.1, 0.25),
                hole_width_range=(0.1, 0.25),
                p=0.3,
            ),
        ]
    )


class TrainTransform:
    """Optionally turns a real crop into a self-blended fake, then augments it.

    Self-blending happens before the degradations so the blend marks get compressed and blurred
    like real fakes would be.
    """

    def __init__(self, preset: Preset = "full", sbi_prob: float = 0.0, seed: int | None = None):
        self.augment = build_augment(preset)
        self.sbi_prob = sbi_prob
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            self.augment.set_random_seed(seed)

    def reseed(self, seed: int) -> None:
        """Give each DataLoader worker its own random stream (called from worker_init_fn)."""
        self.rng = np.random.default_rng(seed)
        self.augment.set_random_seed(seed)

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray | None, label: int
    ) -> tuple[np.ndarray, int]:
        if (
            label == REAL
            and landmarks is not None
            and self.sbi_prob > 0
            and self.rng.random() < self.sbi_prob
        ):
            img, _ = self_blend(img, landmarks, self.rng)
            label = FAKE
        return self.augment(image=img)["image"], label
