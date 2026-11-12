"""Temperature scaling and band cut-offs, fitted on FF++ validation only (S-3, FR-I2).

A confident wrong answer about a real person's video can do real harm, so scores are calibrated
before they are shown. Temperature scaling divides every frame logit by one number T fitted to
minimise log loss on validation frames; it changes confidence, never the ranking (AUC).

    python -m deepfake_detector.calibrate --checkpoint runs/<run>/best.pt

writes runs/<run>/calibration.json, which testing and prediction pick up automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deepfake_detector.metrics import ece, video_scores

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    temperature: float = 1.0
    likely_real_below: float = 0.3
    likely_fake_above: float = 0.7
    video_score: str = "mean"
    val_video_ece_before: float | None = None
    val_video_ece_after: float | None = None
    fitted_on: str = "ffpp val"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls(**json.loads(Path(path).read_text()))

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated fake probability for frame logits."""
        z = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Single temperature minimising binary log loss (Guo et al., 2017)."""
    z = torch.tensor(np.array(logits, np.float64))
    y = torch.tensor(np.array(labels, np.float64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def fit_bands(
    video_probs: np.ndarray, labels: np.ndarray, max_error: float = 0.05, min_width: float = 0.2
) -> tuple[float, float]:
    """Cut-offs so that, on validation, at most `max_error` of fakes land in "likely real" and
    at most `max_error` of reals land in "likely manipulated". Everything between is
    "uncertain".

    The uncertain band is never narrower than `min_width`: validation data that separates
    cleanly would otherwise squeeze it to nothing, and a score near the middle would then get a
    confident label (S-4). Unseen datasets are rarely as clean as FF++ validation.
    """
    p, y = np.asarray(video_probs, float), np.asarray(labels, int)
    real_below = float(np.quantile(p[y == 1], max_error))
    fake_above = float(np.quantile(p[y == 0], 1 - max_error))
    if fake_above - real_below < min_width:
        mid = float(np.clip((real_below + fake_above) / 2, min_width / 2, 1 - min_width / 2))
        real_below, fake_above = mid - min_width / 2, mid + min_width / 2
    return real_below, fake_above


def calibrate_frames(frames: pd.DataFrame, video_score: str = "mean") -> Calibration:
    """Fit temperature and bands on a validation frame table (video_id, label, logit)."""
    t = fit_temperature(frames["logit"].to_numpy(), frames["label"].to_numpy())
    before = video_scores(frames.assign(prob=Calibration(1.0).apply(frames["logit"])), video_score)  # type: ignore[arg-type]
    cal = Calibration(temperature=t, video_score=video_score)
    after = video_scores(frames.assign(prob=cal.apply(frames["logit"])), video_score)  # type: ignore[arg-type]
    lo, hi = fit_bands(after["score"].to_numpy(), after["label"].to_numpy())
    cal.likely_real_below, cal.likely_fake_above = lo, hi
    cal.val_video_ece_before = ece(before["label"].to_numpy(), before["score"].to_numpy())
    cal.val_video_ece_after = ece(after["label"].to_numpy(), after["score"].to_numpy())
    return cal


def main(argv: list[str] | None = None) -> None:
    from deepfake_detector.evaluate import score_split

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: calibration.json next to it")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frames, cfg = score_split(args.checkpoint, split="val")
    cal = calibrate_frames(frames, cfg["train"]["val"]["video_score"])
    out = args.out or args.checkpoint.parent / "calibration.json"
    cal.save(out)
    log.info(
        "temperature %.3f; val video ECE %.4f -> %.4f; bands <%.3f real, >%.3f fake; saved %s",
        cal.temperature,
        cal.val_video_ece_before,
        cal.val_video_ece_after,
        cal.likely_real_below,
        cal.likely_fake_above,
        out,
    )


if __name__ == "__main__":
    main()
