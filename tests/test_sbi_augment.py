import numpy as np
import pytest

from deepfake_detector.data.augment import TrainTransform, build_augment
from deepfake_detector.data.indexers import FAKE, REAL
from deepfake_detector.data.sbi import hull_mask, self_blend


def face_landmarks(size: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    c, r = size / 2, size / 4
    return np.stack([c + r * np.cos(t), c + r * np.sin(t)], axis=1).astype(np.float32)


def test_hull_mask_covers_face_only() -> None:
    mask = hull_mask((64, 64), face_landmarks())
    assert mask[32, 32] == 1.0
    assert mask[0, 0] == 0.0


def test_self_blend_changes_face_not_corners() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out, mask = self_blend(img, face_landmarks(), rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask[32, 32] > 0.2
    assert np.abs(out[32, 32].astype(int) - img[32, 32].astype(int)).sum() > 0
    np.testing.assert_array_equal(out[0, 0], img[0, 0])


def test_self_blend_is_reproducible() -> None:
    img = np.full((64, 64, 3), 128, np.uint8)
    a, _ = self_blend(img, face_landmarks(), np.random.default_rng(1))
    b, _ = self_blend(img, face_landmarks(), np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("preset", ["basic", "full"])
def test_build_augment_keeps_shape(preset: str) -> None:
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = build_augment(preset)(image=img)["image"]  # type: ignore[arg-type]
    assert out.shape == img.shape and out.dtype == np.uint8


def test_build_augment_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        build_augment("nope")  # type: ignore[arg-type]


def test_train_transform_sbi_relabels_real_only() -> None:
    img = np.full((64, 64, 3), 128, np.uint8)
    t = TrainTransform("basic", sbi_prob=1.0, seed=0)
    assert t(img, face_landmarks(), REAL)[1] == FAKE
    assert t(img, face_landmarks(), FAKE)[1] == FAKE
    assert t(img, None, REAL)[1] == REAL
    assert TrainTransform("basic", sbi_prob=0.0)(img, face_landmarks(), REAL)[1] == REAL
