"""Openly licensed deepfake datasets that need no access request, and their importers.

WildDeepfake (Zi et al., ACM MM 2020; Apache-2.0, Hugging Face xingjunm/WildDeepfake)
    707 real and fake videos collected from the internet, shipped as face-crop frame sequences,
    one .tar.gz per video under {real,fake}_{train,test}/. Each archive is downloaded, a few
    evenly spaced frames per sequence are kept, and the archive is deleted.
    Layout made here: data/raw/wilddeepfake/{train,test}/{real,fake}/<video>_<seq>/<frame>.png

OpenForensics (Le et al., ICCV 2021; CC BY 4.0, zenodo.org/record/5528418)
    In-the-wild photos with several faces each, every face labelled real or fake (GAN face
    swaps). Each labelled face is cut out with a margin and saved as its own image, so the label
    belongs to exactly one face.
    Layout made here: data/raw/openforensics/{train,val,test}/{real,fake}/<photo>_<k>.jpg

Archives are read in memory and every output path is built from cleaned parts, so nothing is
written under a name an archive chose.

    python -m deepfake_detector.data.open_sets wilddeepfake --frames 6
    python -m deepfake_detector.data.open_sets openforensics --keep-every 3
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

log = logging.getLogger(__name__)

WDF_REPO = "xingjunm/WildDeepfake"
OF_URL = "https://zenodo.org/api/records/5528418/files/{name}/content"
OF_PARTS = {
    "train": (["Train_part_1.zip", "Train_part_2.zip", "Train_part_3.zip", "Train_part_4.zip",
               "Train_part_5.zip"], "Train_poly.json"),
    "val": (["Val.zip"], "Val_poly.json"),
    "test": (["Test-Dev_part_1.zip", "Test-Dev_part_2.zip"], "Test-Dev_poly.json"),
}  # fmt: skip
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def clean(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in "_-")


def keep_id(key: str, every: int) -> bool:
    """Deterministic 1-in-`every` subsample by a stable hash of the id."""
    return every <= 1 or zlib.crc32(key.encode()) % every == 0


def _decodes(data: bytes) -> bool:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None


# ----------------------------------------------------------------------------- WildDeepfake


def unpack_wdf_tar(tar_path: Path, out_root: Path, split: str, kind: str, frames: int = 6) -> int:
    """Keep `frames` evenly spaced frames of every face sequence in one WildDeepfake archive."""
    from deepfake_detector.data.extract import sample_indices

    if split not in ("train", "test") or kind not in ("real", "fake"):
        raise ValueError("split must be train or test, kind real or fake")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    with tarfile.open(tar_path) as tf:
        for m in tf:
            parts = [
                p for p in PurePosixPath(m.name.replace("\\", "/")).parts if p not in (".", "")
            ]
            if (
                not m.isfile()
                or len(parts) < 4
                or PurePosixPath(parts[-1]).suffix.lower() not in (IMAGE_SUFFIXES)
            ):
                continue
            video, seq = clean(parts[0]), clean(parts[-2])
            if video and seq:
                groups[(video, seq)].append(m.name)
    wanted: dict[str, Path] = {}
    for (video, seq), names in groups.items():
        names.sort(key=lambda n: (len(PurePosixPath(n).stem), PurePosixPath(n).stem))
        for i in sample_indices(len(names), frames):
            stem = clean(PurePosixPath(names[i]).stem) or str(i)
            wanted[names[i]] = out_root / split / kind / f"{video}_{seq}" / f"{stem}.png"
    n = 0
    with tarfile.open(tar_path) as tf:  # second pass: gzip archives read front to back
        for m in tf:
            dest = wanted.get(m.name)
            if dest is None:
                continue
            f = tf.extractfile(m)
            data = f.read() if f is not None else b""
            if not data or not _decodes(data):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if PurePosixPath(m.name).suffix.lower() == ".png":
                dest.write_bytes(data)
            else:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cv2.imwrite(str(dest), img)
            n += 1
    return n


def fetch_wilddeepfake(raw: Path, frames: int = 6, workers: int = 4) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    out = raw / "wilddeepfake"
    done_dir = out / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    tmp = raw.parent / "downloads" / "wilddeepfake"
    todo = []
    for name in HfApi().list_repo_files(WDF_REPO, repo_type="dataset"):
        parts = PurePosixPath(name).parts
        if len(parts) != 3 or not name.endswith(".tar.gz"):
            continue
        kind, _, split = parts[1].partition("_")
        base = f"{clean(parts[1])}_{clean(parts[2].split('.')[0])}"
        marker = done_dir / f"{base}_f{frames}"
        legacy = frames == 6 and (done_dir / base).exists()  # first import kept 6 frames
        if not marker.exists() and not legacy:
            todo.append((name, split, kind, marker))
    log.info("WildDeepfake: %d archives to fetch", len(todo))

    def one(item: tuple[str, str, str, Path]) -> int:
        name, split, kind, marker = item
        path = Path(hf_hub_download(WDF_REPO, name, repo_type="dataset", local_dir=str(tmp)))
        try:
            n = unpack_wdf_tar(path, out, split, kind, frames)
        finally:
            path.unlink(missing_ok=True)
        marker.write_text(str(n))
        return n

    total = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, item) for item in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                total += fut.result()
            except Exception as e:  # one bad archive must not stop the rest; rerun retries it
                log.warning("archive failed: %s", e)
            if i % 50 == 0:
                log.info("WildDeepfake %d/%d archives", i, len(todo))
    return total


# ----------------------------------------------------------------------------- OpenForensics


def crop_box(img: np.ndarray, bbox: list[float], margin: float = 0.5) -> np.ndarray:
    """Cut out a COCO [x, y, w, h] box grown by `margin` of its size on every side."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(x - bw * margin)), max(0, int(y - bh * margin))
    x1, y1 = min(w, int(x + bw * (1 + margin))), min(h, int(y + bh * (1 + margin)))
    return img[y0:y1, x0:x1]


def unpack_openforensics(
    zips: list[Path],
    labels_json: Path,
    out_root: Path,
    split: str,
    keep_every: int = 1,
    min_side: int = 48,
    margin: float = 0.5,
) -> int:
    """Save every labelled face of the kept photos as its own image under real/ or fake/."""
    if split not in OF_PARTS:
        raise ValueError(f"split must be one of {sorted(OF_PARTS)}")
    d = json.loads(labels_json.read_text())
    fake_ids = {c["id"] for c in d["categories"] if c["name"].lower() == "fake"}
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    by_stem = {PurePosixPath(im["file_name"]).stem: im["id"] for im in d["images"]}
    n = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                name = PurePosixPath(info.filename.replace("\\", "/"))
                stem = clean(name.stem)
                if info.is_dir() or name.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if stem not in by_stem or not keep_id(stem, keep_every):
                    continue
                faces = anns.get(by_stem[stem], [])
                if not faces:
                    continue
                img = cv2.imdecode(np.frombuffer(zf.read(info), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                for i, a in enumerate(faces):
                    if min(a["bbox"][2], a["bbox"][3]) < min_side:
                        continue
                    kind = "fake" if a["category_id"] in fake_ids else "real"
                    dest = out_root / split / kind / f"{stem}_{i}.jpg"
                    if not dest.exists():
                        crop = crop_box(img, a["bbox"], margin)
                        if crop.size == 0:
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n += 1
    return n


def import_openforensics(
    downloads: Path, raw: Path, keep_every: dict[str, int], delete: bool = True
) -> int:
    out = raw / "openforensics"
    total = 0
    for split, (zips, labels) in OF_PARTS.items():
        marker = out / f".{split}.every{keep_every[split]}.done"
        if marker.exists():
            total += int(marker.read_text())
            continue
        paths = [downloads / z for z in zips]
        missing = [p.name for p in [*paths, downloads / labels] if not p.exists()]
        if missing:
            log.warning("OpenForensics %s: missing %s, skipped", split, missing)
            continue
        n = unpack_openforensics(paths, downloads / labels, out, split, keep_every[split])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(n))
        total += n
        log.info("OpenForensics %s: %d faces", split, n)
        if delete:
            for p in paths:
                p.unlink(missing_ok=True)
    return total


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("dataset", choices=["wilddeepfake", "openforensics"])
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--downloads", type=Path, default=Path("data/downloads/openforensics"))
    p.add_argument("--frames", type=int, default=6, help="WildDeepfake frames kept per sequence")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--keep-every", type=int, default=3, help="OpenForensics: 1 in k photos")
    p.add_argument("--keep", action="store_true", help="keep the OpenForensics zips")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dataset == "wilddeepfake":
        n = fetch_wilddeepfake(args.raw, args.frames, args.workers)
    else:
        every = {"train": args.keep_every, "val": args.keep_every, "test": args.keep_every}
        n = import_openforensics(args.downloads, args.raw, every, delete=not args.keep)
    log.info("%s: %d images written", args.dataset, n)


if __name__ == "__main__":
    main()
