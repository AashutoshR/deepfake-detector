"""Evaluation on every configured test set: AUC, EER, ECE, ROC, robustness (FR-E1 to FR-E9).

For now this holds frame prediction, shared with training. The full test command comes next.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_frames(
    model: nn.Module, loader: DataLoader, device: torch.device, bf16: bool = True
) -> pd.DataFrame:
    """Fake probability for every frame in the loader: video_id, label, method, logit, prob."""
    model.eval()
    rows: dict[str, list] = {"video_id": [], "label": [], "method": [], "logit": []}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logit = model(x)["logit"].float()
        rows["video_id"] += list(batch["video_id"])
        rows["label"] += batch["label"].int().tolist()
        rows["method"] += list(batch["method"])
        rows["logit"] += logit.cpu().tolist()
    df = pd.DataFrame(rows)
    df["prob"] = torch.sigmoid(torch.tensor(df["logit"].to_numpy())).numpy()
    return df
