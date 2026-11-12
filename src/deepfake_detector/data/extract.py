"""Frame extraction: N evenly spaced frames per video (FR-D1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_indices(n_frames: int, k: int) -> list[int]:
    """Return up to k evenly spaced frame indices in [0, n_frames).

    Short videos give fewer than k indices instead of repeating frames.
    """
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    # Centre each sample in its segment so the first and last frames are not favoured.
    step = n_frames / k
    return [min(n_frames - 1, int(step * i + step / 2)) for i in range(k)]


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8 (H, W, 3)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def count_frames(path: str | Path) -> int:
    """Count frames by decoding; container metadata is unreliable for some datasets."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def read_frames(path: str | Path, k: int) -> list[tuple[int, np.ndarray]]:
    """Read k evenly spaced frames as (frame_index, RGB uint8) pairs.

    Decodes sequentially and keeps only the wanted frames. Slower than seeking, but gives the
    exact same frames on every platform and codec.
    """
    wanted = set(sample_indices(count_frames(path), k))
    if not wanted:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    out: list[tuple[int, np.ndarray]] = []
    last = max(wanted)
    idx = 0
    try:
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, bgr = cap.retrieve()
                if ok:
                    out.append((idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idx += 1
    finally:
        cap.release()
    return out


def frame_files(folder: str | Path) -> list[Path]:
    """Image files of a frame-sequence folder, in natural order (2.png before 10.png)."""

    def key(p: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return (int(digits) if digits else -1, p.name)

    return sorted((p for p in Path(folder).iterdir() if is_image(p)), key=key)


def read_media(path: str | Path, k: int) -> list[tuple[int, np.ndarray]]:
    """Read an image (as frame 0), k frames of a video, or k frames of a folder of images
    (datasets such as WildDeepfake ship sequences as frame folders)."""
    if Path(path).is_dir():
        files = frame_files(path)
        return [(i, read_image(files[i])) for i in sample_indices(len(files), k)]
    if is_image(path):
        return [(0, read_image(path))]
    return read_frames(path, k)
