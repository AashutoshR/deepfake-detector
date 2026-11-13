import numpy as np
import pytest

from deepfake_detector.data.faces import FaceDetection, crop_face, iou, select_main_face


def det(box: tuple[float, float, float, float]) -> FaceDetection:
    return FaceDetection(box=box, score=0.99, landmarks=np.zeros((68, 2), np.float32))


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_select_main_face_largest_then_tracks() -> None:
    small, big = det((0, 0, 10, 10)), det((50, 50, 100, 100))
    assert select_main_face([small, big]) is big
    # When the previous face overlaps the small one, keep following it.
    assert select_main_face([small, big], prev_box=(1, 1, 11, 11)) is small
    # No overlap with the previous face: fall back to the largest.
    assert select_main_face([small, big], prev_box=(200, 200, 210, 210)) is big
    assert select_main_face([]) is None


def test_crop_face_size_and_landmarks() -> None:
    rgb = np.zeros((200, 300, 3), np.uint8)
    rgb[50:150, 100:200] = 255
    lm = np.array([[150.0, 100.0], [100.0, 50.0]], np.float32)
    crop, lm_out = crop_face(rgb, (100, 50, 200, 150), lm, scale=1.0, size=50)
    assert crop.shape == (50, 50, 3)
    assert crop.mean() > 250  # the white box fills the crop at scale 1.0
    assert lm_out is not None
    np.testing.assert_allclose(lm_out, [[25.0, 25.0], [0.0, 0.0]], atol=0.5)


def test_crop_face_pads_outside_image() -> None:
    rgb = np.full((100, 100, 3), 255, np.uint8)
    crop, lm_out = crop_face(rgb, (0, 0, 50, 50), scale=2.0, size=100)
    assert crop.shape == (100, 100, 3)
    assert lm_out is None
    assert crop[:20, :20].max() == 0  # top-left corner lies outside the image
    assert crop[60:, 60:].min() == 255


def test_sfd_detector_turns_off_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """face-alignment 1.5+ compiles with Triton, which Windows lacks; compile must be off."""
    import sys
    import types

    seen: dict[str, object] = {}

    class FakeFA:
        def __init__(
            self, landmarks_type: int, face_detector: str = "sfd", device: str = "cuda",
            compile: bool = True,
        ) -> None:  # fmt: skip
            seen.update(face_detector=face_detector, device=device, compile=compile)

    fake = types.SimpleNamespace(FaceAlignment=FakeFA, LandmarksType=types.SimpleNamespace(TWO_D=1))
    monkeypatch.setitem(sys.modules, "face_alignment", fake)
    from deepfake_detector.data.faces import SFDDetector

    SFDDetector(device="cpu")
    assert seen == {"face_detector": "sfd", "device": "cpu", "compile": False}
