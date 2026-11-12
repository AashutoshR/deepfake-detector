"""Loading a trained detector and scoring frames, shared by training, calibration and testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from deepfake_detector.models import Detector, build_model


def load_detector(
    checkpoint: str | Path, device: torch.device | None = None
) -> tuple[Detector, dict[str, Any]]:
    """Rebuild the model from the config stored in a checkpoint and load its weights.

    Returns (model in eval mode, full run config as a dict).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg: dict[str, Any] = ckpt["config"]
    model_cfg = {**cfg["model"], "grad_checkpointing": False}
    model = build_model(model_cfg, pretrained_weights=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_frames(
    model: nn.Module, loader: DataLoader, device: torch.device, bf16: bool = True
) -> pd.DataFrame:
    """Score every frame in the loader: video_id, label, method, logit, prob."""
    model.eval()
    rows: dict[str, list[Any]] = {"video_id": [], "label": [], "method": [], "logit": []}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logit = model(x)["logit"].float()
        rows["video_id"] += list(batch["video_id"])
        rows["label"] += batch["label"].int().tolist()
        rows["method"] += list(batch["method"])
        rows["logit"] += logit.cpu().tolist()
    df = pd.DataFrame(rows)
    df["prob"] = torch.sigmoid(torch.tensor(df["logit"].to_numpy(dtype="float64"))).numpy()
    return df
