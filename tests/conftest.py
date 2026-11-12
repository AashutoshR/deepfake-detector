from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_detector.data.faces import FaceDetection


def write_video(path: Path, n_frames: int, size: tuple[int, int] = (160, 120)) -> None:
    """Write a tiny synthetic video whose frame i has mean brightness about 2 * i."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), min(255, 2 * i), np.uint8))
    writer.release()


class FixedDetector:
    """Returns one face in the middle of every frame, or none if `empty`."""

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    def detect(self, rgb: np.ndarray) -> list[FaceDetection]:
        if self.empty:
            return []
        h, w = rgb.shape[:2]
        box = (w * 0.3, h * 0.3, w * 0.7, h * 0.7)
        lm = np.tile(np.array([[w / 2, h / 2]], np.float32), (68, 1))
        return [FaceDetection(box=box, score=0.99, landmarks=lm)]


@pytest.fixture
def video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    write_video(p, 40)
    return p
