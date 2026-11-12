"""Command-line prediction: probability, band and per-frame scores (FR-I1, FR-I2, FR-I6).

Usage: predict --input <video or image> --checkpoint <path>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from enum import Enum

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

    probability: float
    band: Band
    frame_scores: list[float] = field(default_factory=list)
    note: str = NOT_PROOF_NOTE

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["band"] = self.band.value
        return d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="predict", description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a video or image")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    raise NotImplementedError("Inference lands after the first trained checkpoint.")


if __name__ == "__main__":
    main()
