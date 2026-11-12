from pathlib import Path

import pytest

from deepfake_detector.data.extract import (
    count_frames,
    is_image,
    read_frames,
    read_image,
    read_media,
    sample_indices,
)


def test_sample_indices_evenly_spaced() -> None:
    idx = sample_indices(100, 4)
    assert idx == [12, 37, 62, 87]


def test_sample_indices_short_video_returns_all() -> None:
    assert sample_indices(5, 32) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("n", "k"), [(0, 32), (10, 0), (-1, 3)])
def test_sample_indices_empty(n: int, k: int) -> None:
    assert sample_indices(n, k) == []


def test_sample_indices_unique_and_sorted() -> None:
    idx = sample_indices(33, 32)
    assert idx == sorted(set(idx)) and len(idx) == 32


def test_count_and_read_frames(video: Path) -> None:
    assert count_frames(video) == 40
    frames = read_frames(video, 4)
    assert [i for i, _ in frames] == sample_indices(40, 4)
    for _, rgb in frames:
        assert rgb.shape == (120, 160, 3)


def test_read_media_image(tmp_path: Path, video: Path) -> None:
    import cv2

    _, rgb = read_frames(video, 1)[0]
    img_path = tmp_path / "x.png"
    cv2.imwrite(str(img_path), rgb)
    assert is_image(img_path)
    media = read_media(img_path, 32)
    assert len(media) == 1 and media[0][0] == 0
    assert read_image(img_path).shape == rgb.shape


def test_bad_paths_raise(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        count_frames(tmp_path / "missing.mp4")
    with pytest.raises(OSError):
        read_image(tmp_path / "missing.png")
