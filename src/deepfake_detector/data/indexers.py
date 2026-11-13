"""Turn each dataset's folder layout into a list of VideoRecord (one per video).

Expected layouts (as shipped by each dataset's download):

FaceForensics++ (root = data/raw/ffpp)
    original_sequences/youtube/<comp>/videos/000.mp4
    manipulated_sequences/<Method>/<comp>/videos/000_003.mp4
    splits/train.json, val.json, test.json   (official split pairs from the FF++ GitHub repo)

Celeb-DF v2 (root = data/raw/celebdf_v2)
    Celeb-real/id0_0000.mp4
    Celeb-synthesis/id0_id16_0000.mp4
    YouTube-real/00000.mp4
    List_of_testing_videos.txt

DFDC subset (root = data/raw/dfdc), made by scripts/get_data.py dfdc
    videos/<name>.mp4
    metadata.json        {"<name>.mp4": {"label": "REAL" | "FAKE", "original": ...}}

FFHQ real faces for the preview model (root = data/raw/ffhq), made by the preview tool
    real/<id>.jpg        one face photo per file; split by id (90% train, 5% val, 5% test)

DeepFakeFace for the second preview model (root = data/raw/dff), made by the preview tool
    wiki/<xx>/<person>_<birth>_<year>.jpg              real photos from IMDB-WIKI
    insight/ inpainting/ text2img/ <same names>         fakes made from those photos

Any other test set (DeeperForensics-1.0, WildDeepfake, DF40), root = data/raw/<dataset>
    real/**              videos, images, or folders of frames (one folder = one sequence)
    fake/<method>/**     the same; <method> names the fake-making method
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
REAL = 0
FAKE = 1


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    video_id: str  # unique within the dataset, e.g. "Deepfakes/000_003"
    label: int  # 0 real, 1 fake
    method: str  # "real" or the fake-making method
    dataset: str
    split: str  # train | val | test
    compression: str  # c23, c40, or "na"
    persons: tuple[str, ...] = field(default_factory=tuple)  # identities shown, for leak checks


def _ffpp_split_map(splits_dir: Path) -> dict[str, str]:
    """Map each FF++ source id ("000") to its official split."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        pairs = json.loads((splits_dir / f"{split}.json").read_text())
        for a, b in pairs:
            out[a] = split
            out[b] = split
    return out


def index_ffpp(
    root: str | Path,
    compressions: Iterable[str] = ("c23",),
    methods: Iterable[str] = FFPP_METHODS,
    splits: Iterable[str] | None = None,
) -> list[VideoRecord]:
    """Index FaceForensics++ with the official 720 / 140 / 140 split."""
    root = Path(root)
    split_of = _ffpp_split_map(root / "splits")
    keep = set(splits) if splits is not None else None
    records: list[VideoRecord] = []
    for comp in compressions:
        real_dir = root / "original_sequences" / "youtube" / comp / "videos"
        for p in sorted(real_dir.glob("*.mp4")):
            sid = p.stem
            split = split_of.get(sid)
            if split is None or (keep is not None and split not in keep):
                continue
            records.append(
                VideoRecord(p, f"real/{sid}", REAL, "real", "ffpp", split, comp, (f"ffpp:{sid}",))
            )
        for method in methods:
            fake_dir = root / "manipulated_sequences" / method / comp / "videos"
            for p in sorted(fake_dir.glob("*.mp4")):
                target, source = p.stem.split("_")
                split = split_of.get(target)
                if split is None or (keep is not None and split not in keep):
                    continue
                persons = (f"ffpp:{target}", f"ffpp:{source}")
                records.append(
                    VideoRecord(p, f"{method}/{p.stem}", FAKE, method, "ffpp", split, comp, persons)
                )
    return records


def index_celebdf(root: str | Path, only_test: bool = True) -> list[VideoRecord]:
    """Index Celeb-DF v2. Labels come from the folder name, not the list file's numbers."""
    root = Path(root)
    test_list = root / "List_of_testing_videos.txt"
    test_paths = {
        line.split(maxsplit=1)[1].strip().replace("\\", "/")
        for line in test_list.read_text().splitlines()
        if line.strip()
    }
    folders = {"Celeb-real": REAL, "YouTube-real": REAL, "Celeb-synthesis": FAKE}
    records: list[VideoRecord] = []
    for folder, label in folders.items():
        for p in sorted((root / folder).glob("*.mp4")):
            rel = f"{folder}/{p.name}"
            split = "test" if rel in test_paths else "train"
            if only_test and split != "test":
                continue
            if folder == "YouTube-real":
                persons: tuple[str, ...] = (f"celebdf:yt{p.stem}",)
            else:
                persons = tuple(f"celebdf:{t}" for t in p.stem.split("_") if t.startswith("id"))
            method = "real" if label == REAL else "celebdf_synthesis"
            records.append(
                VideoRecord(
                    p, rel.removesuffix(".mp4"), label, method, "celebdf_v2", split, "na", persons
                )
            )
    return records


def index_dfdc(root: str | Path) -> list[VideoRecord]:
    """Index the fixed DFDC subset. A fake shares a person id with the real video it came from."""
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    records = []
    for name, info in sorted(meta.items()):
        path = root / "videos" / name
        label = FAKE if info["label"].upper() == "FAKE" else REAL
        source = Path(info.get("original") or name).stem
        records.append(
            VideoRecord(path, Path(name).stem, label, "dfdc" if label else "real", "dfdc",
                        "test", "na", (f"dfdc:{source}",))
        )  # fmt: skip
    return records


MEDIA_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _media_items(folder: Path) -> list[Path]:
    """Videos, single images, and frame folders (a folder whose files are images) below folder."""
    from deepfake_detector.data.extract import is_image

    items: list[Path] = []
    for d in sorted([folder, *(p for p in folder.rglob("*") if p.is_dir())]):
        files = [p for p in d.iterdir() if p.is_file()]
        images = [p for p in files if is_image(p)]
        if images and len(images) == len(files) and d != folder:
            items.append(d)  # a sequence of frames
            continue
        items += sorted(p for p in files if p.suffix.lower() in MEDIA_SUFFIXES)
        if d == folder:
            items += sorted(images)  # loose single images at the top level
    return items


def index_folders(root: str | Path, dataset: str, split: str = "test") -> list[VideoRecord]:
    """Index a dataset arranged as real/** and fake/<method>/** (see module docstring).

    These test sets publish no identity labels, so each item is its own person for the leak
    check; they are only ever used as test sets.
    """
    root = Path(root)
    records: list[VideoRecord] = []
    for p in _media_items(root / "real") if (root / "real").exists() else []:
        vid = p.relative_to(root).with_suffix("").as_posix()
        records.append(
            VideoRecord(p, vid, REAL, "real", dataset, split, "na", (f"{dataset}:{vid}",))
        )
    fake_root = root / "fake"
    for method_dir in (
        sorted(d for d in fake_root.iterdir() if d.is_dir()) if fake_root.exists() else []
    ):
        for p in _media_items(method_dir):
            vid = p.relative_to(root).with_suffix("").as_posix()
            records.append(
                VideoRecord(
                    p, vid, FAKE, method_dir.name, dataset, split, "na", (f"{dataset}:{vid}",)
                )
            )
    return records


def index_ffhq(root: str | Path) -> list[VideoRecord]:
    """Index FFHQ real photos for the self-blended preview model (real only; every fake is made
    during training). Each photo is its own person and its own one-frame "video"."""
    records = []
    for p in sorted((Path(root) / "real").glob("*.jpg")):
        n = int(p.stem) if p.stem.isdigit() else sum(map(ord, p.stem))
        split = {0: "val", 1: "test"}.get(n % 20, "train")
        vid = f"real/{p.stem}"
        records.append(VideoRecord(p, vid, REAL, "real", "ffhq", split, "na", (f"ffhq:{p.stem}",)))
    return records


DFF_FOLDERS = {"wiki": "real", "insight": "insight", "inpainting": "inpainting",
               "text2img": "text2img"}  # fmt: skip


def dff_split(person: str) -> str:
    """Split DeepFakeFace by person id so no person is in two splits (90 / 5 / 5)."""
    n = int(person) if person.isdigit() else zlib.crc32(person.encode())
    return {0: "val", 1: "test"}.get(n % 20, "train")


def index_dff(root: str | Path) -> list[VideoRecord]:
    """Index DeepFakeFace: real IMDB-WIKI photos and three fakes made from the same photos.

    Every file is <folder>/<xx>/<person>_<birth>_<year>.jpg and the fakes reuse the real photo's
    name, so a real photo and its fakes always share one split.
    """
    root = Path(root)
    records = []
    for folder, method in DFF_FOLDERS.items():
        label = REAL if method == "real" else FAKE
        for p in sorted((root / folder).glob("*/*.jpg")):
            person = p.stem.split("_")[0]
            vid = f"{folder}/{p.parent.name}/{p.stem}"
            records.append(
                VideoRecord(p, vid, label, method, "dff", dff_split(person), "na",
                            (f"dff:{person}",))
            )  # fmt: skip
    return records


INDEXERS: dict[str, Callable[..., list[VideoRecord]]] = {
    "ffpp": index_ffpp,
    "celebdf_v2": index_celebdf,
    "dfdc": index_dfdc,
    "deeperforensics": lambda root: index_folders(root, "deeperforensics"),
    "wilddeepfake": lambda root: index_folders(root, "wilddeepfake"),
    "df40": lambda root: index_folders(root, "df40"),
    "ffhq": index_ffhq,
    "dff": index_dff,
}
