"""Predict whether a video or image is manipulated (FR-I1, FR-I2, FR-I6).

Prints JSON with a calibrated probability, one of three bands (likely real, uncertain, likely
manipulated), per-frame scores and a note that the score is an estimate, not proof.

    predict --input clip.mp4 --checkpoint runs/<run>/best.pt
    predict --input face.jpg --checkpoint runs/<run>/best.pt --heatmap heat.png
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from deepfake_detector.calibrate import Calibration
from deepfake_detector.data.extract import read_media
from deepfake_detector.data.faces import Box, FaceDetector, crop_face, select_main_face
from deepfake_detector.inference import load_detector

NOT_PROOF_NOTE = (
    "This score is an estimate from a statistical model, not proof that the media is real "
    "or manipulated."
)


class Band(str, Enum):
    """Three bands instead of a hard verdict (S-4)."""

    LIKELY_REAL = "likely real"
    UNCERTAIN = "uncertain"
    LIKELY_MANIPULATED = "likely manipulated"


@dataclass(frozen=True)
class BandCutoffs:
    """Cut-offs on the calibrated probability, set on validation data only."""

    likely_real_below: float = 0.3
    likely_fake_above: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.likely_real_below <= self.likely_fake_above <= 1.0:
            raise ValueError("cut-offs must satisfy 0 <= real_below <= fake_above <= 1")


def to_band(probability: float, cutoffs: BandCutoffs | None = None) -> Band:
    """Map a calibrated fake probability to one of the three bands."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    c = cutoffs or BandCutoffs()
    if probability < c.likely_real_below:
        return Band.LIKELY_REAL
    if probability > c.likely_fake_above:
        return Band.LIKELY_MANIPULATED
    return Band.UNCERTAIN


@dataclass
class Prediction:
    """Result shared by the CLI and the API (FR-I3)."""

    probability: float | None
    band: Band | None
    frame_scores: list[float] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    status: str = "ok"  # ok | no_face
    frames_read: int = 0
    faces_found: int = 0
    cutoffs: dict[str, float] = field(default_factory=dict)
    heatmap_frame: int | None = None
    heatmap_png_base64: str | None = None
    elapsed_ms: float = 0.0
    note: str = NOT_PROOF_NOTE

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["band"] = self.band.value if self.band is not None else None
        return d


class Predictor:
    """Loads the model, calibration and face detector once; scores media files.

    Runs on the GPU when there is one and on the CPU otherwise (NFR-5).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        detector: FaceDetector | None = None,
        frames: int = 32,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, cfg = load_detector(checkpoint, self.device)
        cal_path = Path(checkpoint).parent / "calibration.json"
        self.calibration = Calibration.load(cal_path) if cal_path.exists() else Calibration()
        self.calibrated = cal_path.exists()
        faces = cfg["data"]["faces"]
        self.crop_scale = float(faces["crop_scale"])
        self.size = int(cfg["model"].get("img_size") or faces["size"])
        self.frames = frames
        self.is_vit = cfg["model"].get("arch", "vit") == "vit"
        self.checkpoint_name = f"{Path(checkpoint).parent.name}/{Path(checkpoint).name}"
        self.model_note: str | None = None
        if cfg.get("train", {}).get("dataset") == "ffhq":
            from deepfake_detector.preview import PREVIEW_NOTE

            self.model_note = PREVIEW_NOTE
        if detector is None:
            from deepfake_detector.data.faces import SFDDetector

            detector = SFDDetector(device=self.device.type)
        self.detector = detector

    @torch.no_grad()
    def warm_up(self) -> None:
        """Run the detector and model once so the first real request is not slow (GPU kernels
        are picked on first use)."""
        self.detector.detect(np.full((self.size * 2, self.size * 2, 3), 128, np.uint8))
        x = torch.zeros(1, 3, self.size, self.size, device=self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            self.model(x)

    @property
    def cutoffs(self) -> BandCutoffs:
        c = self.calibration
        return BandCutoffs(c.likely_real_below, c.likely_fake_above)

    def _crops(self, path: str | Path) -> tuple[list[int], list[np.ndarray], int]:
        media = read_media(path, self.frames)
        idxs: list[int] = []
        crops: list[np.ndarray] = []
        prev_box: Box | None = None
        for idx, rgb in media:
            face = select_main_face(self.detector.detect(rgb), prev_box)
            if face is None:
                continue
            prev_box = face.box
            crop, _ = crop_face(rgb, face.box, scale=self.crop_scale, size=self.size)
            idxs.append(idx)
            crops.append(crop)
        return idxs, crops, len(media)

    @torch.no_grad()
    def predict(self, path: str | Path, heatmap: bool = True) -> Prediction:
        t0 = time.perf_counter()
        idxs, crops, n_read = self._crops(path)
        cut = self.cutoffs
        cutoffs = {
            "likely_real_below": cut.likely_real_below,
            "likely_fake_above": cut.likely_fake_above,
        }
        if not crops:
            return Prediction(
                None,
                None,
                status="no_face",
                frames_read=n_read,
                cutoffs=cutoffs,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(self.device)
        logits = []
        bf16 = self.device.type == "cuda"
        for chunk in x.split(32):
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                logits.append(self.model(chunk)["logit"].float().cpu())
        probs = self.calibration.apply(torch.cat(logits).numpy())
        prob = float(_video_score(probs, self.calibration.video_score))

        result = Prediction(
            probability=prob,
            band=to_band(prob, cut),
            frame_scores=[round(float(p), 4) for p in probs],
            frames=idxs,
            frames_read=n_read,
            faces_found=len(crops),
            cutoffs=cutoffs,
        )
        if heatmap and self.is_vit:
            from deepfake_detector.explain import attention_rollout, overlay

            top = int(np.argmax(probs))  # the frame the model finds most suspicious
            heat = attention_rollout(self.model, x[top : top + 1])[0]
            result.heatmap_frame = idxs[top]
            result.heatmap_png_base64 = png_base64(overlay(crops[top], heat))
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def make_test_fake(
        self, path: str | Path, seed: int | None = None, max_side: int = 1024
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Self-blend the main face of a photo with a changed copy of itself (same person, no
        identity change) to get a synthetic fake for trying the detector.

        Returns (original, blended) as RGB arrays, or None when no face is found.
        """
        from deepfake_detector.data.sbi import self_blend
        from deepfake_detector.preview import shrink

        media = read_media(path, 1)
        if not media:
            return None
        rgb = shrink(media[0][1], max_side)
        face = select_main_face(self.detector.detect(rgb), None)
        if face is None or face.landmarks is None:
            return None
        blended, _ = self_blend(rgb, face.landmarks, np.random.default_rng(seed))
        return rgb, blended


def _video_score(probs: np.ndarray, how: str, k: int = 8) -> float:
    if how == "median":
        return float(np.median(probs))
    if how == "topk":
        return float(np.sort(probs)[-k:].mean())
    return float(np.mean(probs))


def png_base64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to a video or image")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    parser.add_argument("--frames", type=int, default=32, help="frames sampled from a video")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--heatmap", type=Path, default=None, help="save a heatmap PNG here")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictor = Predictor(args.checkpoint, device=args.device, frames=args.frames)
    result = predictor.predict(args.input, heatmap=args.heatmap is not None)
    out: dict[str, Any] = result.to_dict()
    png = out.pop("heatmap_png_base64")
    if args.heatmap is not None and png:
        args.heatmap.write_bytes(base64.b64decode(png))
        out["heatmap_saved_to"] = str(args.heatmap)
    if not predictor.calibrated:
        out["warning"] = "no calibration.json next to the checkpoint; scores are uncalibrated"
    if predictor.model_note:
        out["model_note"] = predictor.model_note
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
