"""Face detection, landmarks, main-face selection and cropping (FR-D2 to FR-D4).

One detector (face-alignment with SFD) is used for every dataset so preprocessing is identical
everywhere (FR-D6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceDetection:
    """One detected face. box is (x0, y0, x1, y1) in pixels; landmarks is (68, 2)."""

    box: Box
    score: float
    landmarks: np.ndarray

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class FaceDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[FaceDetection]: ...


class SFDDetector:
    """face-alignment SFD detector plus 68-point 2D landmarks."""

    def __init__(self, device: str = "cuda", min_score: float = 0.9) -> None:
        import face_alignment

        self.min_score = min_score
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, face_detector="sfd", device=device
        )

    def detect(self, rgb: np.ndarray) -> list[FaceDetection]:
        landmarks, _, boxes = self._fa.get_landmarks_from_image(rgb, return_bboxes=True)
        if landmarks is None or boxes is None:
            return []
        out = []
        for lm, b in zip(landmarks, boxes, strict=False):
            score = float(b[4]) if len(b) > 4 else 1.0
            if score < self.min_score:
                continue
            box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            out.append(FaceDetection(box=box, score=score, landmarks=np.asarray(lm, np.float32)))
        return out


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_main_face(
    detections: list[FaceDetection], prev_box: Box | None = None, min_iou: float = 0.3
) -> FaceDetection | None:
    """Pick the main face: follow the previous frame's face if it overlaps, else the largest."""
    if not detections:
        return None
    if prev_box is not None:
        best = max(detections, key=lambda d: iou(d.box, prev_box))
        if iou(best.box, prev_box) >= min_iou:
            return best
    return max(detections, key=lambda d: d.area)


def crop_face(
    rgb: np.ndarray,
    box: Box,
    landmarks: np.ndarray | None = None,
    scale: float = 1.3,
    size: int = 224,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Square crop around the box enlarged by `scale`, resized to size x size (FR-D4).

    Parts of the crop outside the image are zero-padded. Landmarks are mapped into crop
    coordinates when given.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side_px = max(1, int(round(max(x1 - x0, y1 - y0) * scale)))
    left, top = int(round(cx - side_px / 2)), int(round(cy - side_px / 2))

    h, w = rgb.shape[:2]
    canvas = np.zeros((side_px, side_px, 3), dtype=rgb.dtype)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side_px), min(h, top + side_px)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = rgb[sy0:sy1, sx0:sx1]

    interp = cv2.INTER_AREA if side_px > size else cv2.INTER_CUBIC
    crop = cv2.resize(canvas, (size, size), interpolation=interp)

    lm_out = None
    if landmarks is not None:
        offset = np.array([left, top], np.float32)
        lm_out = ((landmarks - offset) * (size / side_px)).astype(np.float32)
    return crop, lm_out
