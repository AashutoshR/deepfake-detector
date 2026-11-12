import pytest

from deepfake_detector.predict import (
    NOT_PROOF_NOTE,
    Band,
    BandCutoffs,
    Prediction,
    build_parser,
    to_band,
)


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, Band.LIKELY_REAL),
        (0.29, Band.LIKELY_REAL),
        (0.3, Band.UNCERTAIN),
        (0.5, Band.UNCERTAIN),
        (0.7, Band.UNCERTAIN),
        (0.71, Band.LIKELY_MANIPULATED),
        (1.0, Band.LIKELY_MANIPULATED),
    ],
)
def test_to_band_default_cutoffs(p: float, expected: Band) -> None:
    assert to_band(p) is expected


def test_to_band_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        to_band(1.5)


def test_cutoffs_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        BandCutoffs(likely_real_below=0.8, likely_fake_above=0.2)


def test_prediction_always_carries_note() -> None:
    d = Prediction(probability=0.9, band=Band.LIKELY_MANIPULATED).to_dict()
    assert d["note"] == NOT_PROOF_NOTE
    assert d["band"] == "likely manipulated"


def test_parser_requires_input_and_checkpoint() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
