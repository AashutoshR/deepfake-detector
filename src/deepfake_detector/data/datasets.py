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


def add_sbi_copies(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add a self-blended fake copy of every real crop, made the same way every time.

    For a real-only dataset (the FFHQ preview), validation and test need fakes too; these copies
    give a fixed, repeatable set so model selection and calibration are stable across epochs.
    """
    real = df[df["label"] == 0]
    fakes = real.assign(
        label=1,
        method="sbi",
        video_id=real["video_id"] + "#sbi",
        sbi_seed=[seed * 1_000_003 + i for i in range(len(real))],
    )
    out = pd.concat([df.assign(sbi_seed=-1), fakes], ignore_index=True)
    out["sbi_seed"] = out["sbi_seed"].astype("int64")
    return out


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
        method = str(row["method"])
        if int(row.get("sbi_seed", -1)) >= 0 and landmarks is not None:
            from deepfake_detector.data.sbi import self_blend

            img, _ = self_blend(img, landmarks, np.random.default_rng(int(row["sbi_seed"])))
        if self.transform is not None:
            new_img, new_label = self.transform(img, landmarks, label)
            if new_label != label:
                method = "sbi"  # a real crop turned into a self-blended fake
            img, label = new_img, new_label
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": row["video_id"],
            "method": method,
        }


def balanced_sampler(df: pd.DataFrame, seed: int = 0) -> WeightedRandomSampler:
    """Sample real and fake crops equally often (FF++ has 4 fakes per real video)."""
    counts = df["label"].value_counts()
    weights = np.array(df["label"].map(lambda y: 1.0 / counts[y]), dtype=np.float64)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights.tolist(), len(df), replacement=True, generator=g)
