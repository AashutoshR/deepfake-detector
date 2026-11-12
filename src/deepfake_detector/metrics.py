"""Metrics: frame and video AUC, EER, accuracy, log loss and ECE (FR-E2, FR-E3)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

VideoScore = Literal["mean", "median", "topk"]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC, or NaN when only one class is present."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal error rate: where false-positive and false-negative rates meet."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def ece(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on the fake probability."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    if len(labels) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(probs[sel].mean() - labels[sel].mean())
    return float(total)


def log_loss(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    labels = np.asarray(labels, float)
    p = np.clip(np.asarray(probs, float), eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def video_scores(
    frames: pd.DataFrame, how: VideoScore = "mean", k: int = 8, score_col: str = "prob"
) -> pd.DataFrame:
    """Combine frame scores into one score per video (FR-E3).

    frames needs columns video_id, label and score_col. Returns video_id, label, score.
    """
    g = frames.groupby("video_id")
    if how == "mean":
        s = g[score_col].mean()
    elif how == "median":
        s = g[score_col].median()
    elif how == "topk":
        s = g[score_col].apply(lambda x: x.nlargest(k).mean())
    else:
        raise ValueError(f"unknown video score: {how}")
    return pd.DataFrame({"label": g["label"].first(), "score": s}).reset_index()


def summarize(frames: pd.DataFrame, how: VideoScore = "mean", k: int = 8) -> dict[str, float]:
    """All headline metrics from a frame table with video_id, label and prob columns."""
    videos = video_scores(frames, how, k)
    y, p = videos["label"].to_numpy(), videos["score"].to_numpy()
    return {
        "frame_auc": auc(frames["label"].to_numpy(), frames["prob"].to_numpy()),
        "video_auc": auc(y, p),
        "video_eer": eer(y, p),
        "video_acc": float(np.mean((p >= 0.5) == y)) if len(y) else float("nan"),
        "video_log_loss": log_loss(y, p) if len(y) else float("nan"),
        "video_ece": ece(y, p),
        "n_videos": float(len(videos)),
        "n_frames": float(len(frames)),
    }


def auc_per_method(frames: pd.DataFrame, how: VideoScore = "mean") -> dict[str, float]:
    """Video AUC of each fake method against all real videos (FR-E5)."""
    videos = video_scores(frames, how)
    methods = frames.groupby("video_id")["method"].first()
    videos = videos.join(methods, on="video_id")
    real = videos[videos["label"] == 0]
    out = {}
    for m in sorted(set(videos["method"]) - {"real"}):
        part = pd.concat([real, videos[videos["method"] == m]])
        out[m] = auc(part["label"].to_numpy(), part["score"].to_numpy())
    return out
