"""Preprocess a dataset into face crops plus a Parquet manifest (FR-D1 to FR-D8).

Each video writes its own shard file, so a stopped run resumes where it left off (FR-D8).
Videos with no usable face go to a skip table with a reason (FR-D7).

Example:
    python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp --out data
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from deepfake_detector.data.extract import read_media
from deepfake_detector.data.faces import Box, FaceDetector, crop_face, select_main_face
from deepfake_detector.data.indexers import INDEXERS, VideoRecord
from deepfake_detector.data.manifest import CROP_COLUMNS, SKIP_COLUMNS, merge_shards

log = logging.getLogger(__name__)


def shard_name(video_id: str) -> str:
    return video_id.replace("/", "__").replace("\\", "__") + ".parquet"


def process_video(
    rec: VideoRecord,
    detector: FaceDetector,
    out_root: Path,
    frames: int = 32,
    scale: float = 1.3,
    size: int = 224,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect, crop and save faces for one video. Returns (crop rows, skip rows)."""
    base = {
        "dataset": rec.dataset,
        "video_id": rec.video_id,
        "split": rec.split,
        "label": rec.label,
        "method": rec.method,
    }
    try:
        media = read_media(rec.path, frames)
    except OSError as e:
        skip = {**base, "reason": f"read_error: {e}", "frames_read": 0}
        return pd.DataFrame(columns=CROP_COLUMNS), pd.DataFrame([skip], columns=SKIP_COLUMNS)

    crop_dir = out_root / "crops" / rec.dataset / rec.video_id
    rows = []
    prev: Box | None = None
    for idx, rgb in media:
        face = select_main_face(detector.detect(rgb), prev)
        if face is None:
            continue
        prev = face.box
        crop, lm = crop_face(rgb, face.box, face.landmarks, scale=scale, size=size)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{idx:05d}.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                **base,
                "frame": idx,
                "compression": rec.compression,
                "crop_path": crop_path.relative_to(out_root).as_posix(),
                "det_score": face.score,
                "landmarks": lm.reshape(-1).tolist() if lm is not None else None,
                "persons": list(rec.persons),
            }
        )
    crops = pd.DataFrame(rows, columns=CROP_COLUMNS)
    if crops.empty:
        reason = "no_frames" if not media else "no_face"
        skip = {**base, "reason": reason, "frames_read": len(media)}
        return crops, pd.DataFrame([skip], columns=SKIP_COLUMNS)
    return crops, pd.DataFrame(columns=SKIP_COLUMNS)


_detector: FaceDetector | None = None


def _init_worker(device: str) -> None:
    global _detector
    from deepfake_detector.data.faces import SFDDetector

    cv2.setNumThreads(1)
    _detector = SFDDetector(device=device)


def _work(rec: VideoRecord, out_root: Path, frames: int, scale: float, size: int) -> str:
    assert _detector is not None
    crops, skips = process_video(rec, _detector, out_root, frames, scale, size)
    man_dir = out_root / "manifests" / rec.dataset
    name = shard_name(rec.video_id)
    if not skips.empty:
        (man_dir / "skips").mkdir(parents=True, exist_ok=True)
        skips.to_parquet(man_dir / "skips" / name, index=False)
        return "skipped"
    (man_dir / "shards").mkdir(parents=True, exist_ok=True)
    crops.to_parquet(man_dir / "shards" / name, index=False)
    return "ok"


def pending(records: Iterable[VideoRecord], out_root: Path) -> list[VideoRecord]:
    """Records not yet processed (no shard and no skip file)."""
    todo = []
    for r in records:
        man_dir = out_root / "manifests" / r.dataset
        name = shard_name(r.video_id)
        if not (man_dir / "shards" / name).exists() and not (man_dir / "skips" / name).exists():
            todo.append(r)
    return todo


def run(
    records: list[VideoRecord],
    out_root: Path,
    frames: int = 32,
    scale: float = 1.3,
    size: int = 224,
    workers: int = 2,
    device: str = "cuda",
) -> dict[str, int]:
    todo = pending(records, out_root)
    log.info("%d of %d videos left to process", len(todo), len(records))
    counts = {"ok": 0, "skipped": 0, "already_done": len(records) - len(todo)}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(device,)) as pool:
        futures = [pool.submit(_work, r, out_root, frames, scale, size) for r in todo]
        for f in tqdm(as_completed(futures), total=len(futures), desc="videos"):
            counts[f.result()] += 1
    return counts


def finalize(dataset: str, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge shards into manifest.parquet and skips into skipped.parquet, with counts logged."""
    man_dir = out_root / "manifests" / dataset
    crops = merge_shards(man_dir / "shards", man_dir / "manifest.parquet")
    skip_files = sorted((man_dir / "skips").glob("*.parquet"))
    skips = (
        pd.concat([pd.read_parquet(p) for p in skip_files], ignore_index=True)
        if skip_files
        else pd.DataFrame(columns=SKIP_COLUMNS)
    )
    skips.to_parquet(man_dir / "skipped.parquet", index=False)
    n_videos = crops["video_id"].nunique() if not crops.empty else 0
    log.info(
        "%s: %d crops from %d videos; %d videos skipped", dataset, len(crops), n_videos, len(skips)
    )
    if not skips.empty:
        log.warning("skip reasons:\n%s", skips["reason"].value_counts().to_string())
    return crops, skips


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(INDEXERS))
    p.add_argument("--raw", required=True, type=Path, help="dataset root folder")
    p.add_argument("--out", default=Path("data"), type=Path)
    p.add_argument("--frames", default=32, type=int)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compression", nargs="+", default=["c23"], help="FF++ only")
    p.add_argument("--splits", nargs="+", default=None, help="FF++ only, e.g. test")
    p.add_argument("--limit", type=int, default=None, help="process only the first N videos")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset == "ffpp":
        records = INDEXERS["ffpp"](args.raw, compressions=args.compression, splits=args.splits)
    else:
        records = INDEXERS[args.dataset](args.raw)
    if args.limit:
        records = records[: args.limit]
    counts = run(records, args.out, args.frames, workers=args.workers, device=args.device)
    log.info("run counts: %s", counts)
    finalize(args.dataset, args.out)


if __name__ == "__main__":
    main()
