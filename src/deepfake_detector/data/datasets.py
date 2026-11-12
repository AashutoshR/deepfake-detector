"""PyTorch dataset reading face crops from the Parquet manifest (FR-D5, FR-T5)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_manifest(
    path: str | Path,
    split: str | None = None,
    leave_out_method: str | None = None,
    compressions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a manifest, optionally filtered to one split, dropping one fake method (FR-T5)."""
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split]
    if leave_out_method is not None:
        df = df[df["method"] != leave_out_method]
    if compressions is not None:
        df = df[df["compression"].isin(list(compressions))]
    return df.reset_index(drop=True)


class FaceCropDataset(Dataset[dict[str, Any]]):
    """Returns dicts with image (C, H, W float tensor), label, video_id and landmarks."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        transform: Callable[[np.ndarray, np.ndarray | None, int], tuple[np.ndarray, int]]
        | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        bgr = cv2.imread(str(self.root / row["crop_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"missing crop: {row['crop_path']}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = row["landmarks"]
        landmarks = None if lm is None else np.asarray(lm, np.float32).reshape(-1, 2)
        label = int(row["label"])
        if self.transform is not None:
            img, label = self.transform(img, landmarks, label)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": row["video_id"],
            "method": row["method"],
        }


def balanced_sampler(df: pd.DataFrame, seed: int = 0) -> WeightedRandomSampler:
    """Sample real and fake crops equally often (FF++ has 4 fakes per real video)."""
    counts = df["label"].value_counts()
    weights = np.array(df["label"].map(lambda y: 1.0 / counts[y]), dtype=np.float64)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights.tolist(), len(df), replacement=True, generator=g)
