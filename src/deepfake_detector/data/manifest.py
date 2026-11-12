"""Manifest schema, merging and split-leak checks (FR-D5, FR-D7).

One row per saved face crop. Videos with no usable face are logged in a separate skip table,
never dropped silently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CROP_COLUMNS = [
    "dataset",
    "video_id",
    "frame",
    "label",
    "method",
    "split",
    "compression",
    "crop_path",
    "det_score",
    "landmarks",  # flat list of 136 floats in crop coordinates
    "persons",  # list of identity ids shown in the video
]
SKIP_COLUMNS = ["dataset", "video_id", "split", "label", "method", "reason", "frames_read"]


class SplitLeakError(ValueError):
    """Raised when a video or a person appears in more than one split."""


def check_no_leaks(df: pd.DataFrame) -> None:
    """Fail if any video id or person appears in more than one split of the same dataset."""
    for dataset, part in df.groupby("dataset"):
        vid_splits = part.groupby("video_id")["split"].nunique()
        bad_videos = vid_splits[vid_splits > 1]
        if not bad_videos.empty:
            raise SplitLeakError(
                f"{dataset}: {len(bad_videos)} videos in several splits, "
                f"e.g. {list(bad_videos.index[:5])}"
            )
        persons = part[["persons", "split"]].explode("persons").dropna().drop_duplicates()
        per_person = persons.groupby("persons")["split"].nunique()
        bad_people = per_person[per_person > 1]
        if not bad_people.empty:
            raise SplitLeakError(
                f"{dataset}: {len(bad_people)} people in several splits, "
                f"e.g. {list(bad_people.index[:5])}"
            )


def merge_shards(shard_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Combine per-video shard files into one manifest, check for leaks, and save it."""
    shards = sorted(Path(shard_dir).glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in shards]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CROP_COLUMNS)
    if not df.empty:
        check_no_leaks(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
