"""Get each dataset into data/raw/<dataset> and check it (run after accepting its license).

None of these datasets may be re-shared. This tool never downloads anything without the link
or script you received after your own access request; it only runs, unpacks and checks.

    python scripts/get_data.py disk
    python scripts/get_data.py ffpp --script faceforensics_download_v4.py
    python scripts/get_data.py celebdf --zip Celeb-DF-v2.zip
    python scripts/get_data.py dfdc --zip dfdc_train_part_0.zip --n 1000
    python scripts/get_data.py folder --dataset wilddeepfake --real <dir> --fake <dir> --method wild
    python scripts/get_data.py check
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from deepfake_detector.data.indexers import FFPP_METHODS, INDEXERS

FFPP_SPLITS_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"
)
NEEDED_GB = 100  # whole project, from the plan
ROUGH_GB = {"ffpp_c23": 10, "ffpp_c40": 3, "celebdf_v2": 10, "dfdc_part": 10, "crops": 20}


# ----------------------------------------------------------------------------- helpers


def free_gb(path: str | Path = ".") -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def _safe_target(dest: Path, member: str) -> Path | None:
    """Where a zip member may be written, or None if it would escape dest (zip slip)."""
    parts = PurePosixPath(member.replace("\\", "/")).parts
    if not parts or parts[0] in ("/", "") or ".." in parts or ":" in parts[0]:
        return None
    target = (dest / Path(*parts)).resolve()
    return target if target.is_relative_to(dest.resolve()) else None


def safe_extract(
    zf: zipfile.ZipFile,
    dest: Path,
    members: Sequence[str] | None = None,
    rename: Callable[[str], str | None] | None = None,
) -> int:
    """Extract members (default all) into dest, refusing absolute or ../ paths.

    rename maps a member name to its path below dest, or None to skip it.
    Returns the number of files written.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in members if members is not None else zf.namelist():
        if name.endswith("/"):
            continue
        rel = rename(name) if rename else name
        if rel is None:
            continue
        target = _safe_target(dest, rel)
        if target is None:
            raise ValueError(f"unsafe path in archive: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
        written += 1
    return written


def _strip_top_folder(names: Sequence[str], keep: Sequence[str]) -> Callable[[str], str | None]:
    """If every file sits under one top folder (e.g. Celeb-DF-v2/), drop that folder."""
    files = [n.replace("\\", "/") for n in names if not n.endswith("/")]
    tops = {n.split("/", 1)[0] for n in files}
    strip = len(tops) == 1 and not any(t in keep for t in tops) and all("/" in n for n in files)

    def rename(name: str) -> str | None:
        n = name.replace("\\", "/")
        return n.split("/", 1)[1] if strip else n

    return rename


# ----------------------------------------------------------------------------- per dataset


def get_ffpp(
    script: Path,
    out: Path,
    compressions: Sequence[str] = ("c23", "c40"),
    server: str = "EU2",
    run: Callable[[list[str]], object] = lambda cmd: subprocess.run(cmd, check=True),
    fetch: Callable[[str], bytes] = lambda url: urllib.request.urlopen(url, timeout=60).read(),
) -> None:
    """Run the official FF++ download script (it asks you to accept the terms) for the real
    videos and the four methods only, then fetch the official split lists."""
    for comp in compressions:
        for d in ("original", *FFPP_METHODS):
            run([sys.executable, str(script), str(out), "-d", d, "-c", comp, "-t", "videos",
                 "--server", server])  # fmt: skip
    splits = out / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        data = fetch(FFPP_SPLITS_URL.format(s))
        pairs = json.loads(data)
        assert all(len(p) == 2 for p in pairs), f"unexpected format in {s}.json"
        (splits / f"{s}.json").write_bytes(data)


def get_celebdf(zip_path: Path, out: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        keep = ("Celeb-real", "Celeb-synthesis", "YouTube-real")
        return safe_extract(zf, out, names, _strip_top_folder(names, keep))


def get_dfdc_subset(zip_path: Path, out: Path, n: int = 1000, seed: int = 0) -> dict[str, int]:
    """Pick a fixed, seeded subset of one DFDC part: all reals up to n/2, fakes for the rest.

    Writes out/videos/*.mp4 and out/metadata.json for the chosen videos only.
    """
    with zipfile.ZipFile(zip_path) as zf:
        meta_names = [m for m in zf.namelist() if m.replace("\\", "/").endswith("metadata.json")]
        if len(meta_names) != 1:
            raise ValueError(f"expected one metadata.json in {zip_path}, found {len(meta_names)}")
        meta: dict[str, dict] = json.loads(zf.read(meta_names[0]))
        folder = (
            meta_names[0].replace("\\", "/").rsplit("/", 1)[0] + "/" if "/" in meta_names[0] else ""
        )
        rng = np.random.default_rng(seed)
        real = sorted(k for k, v in meta.items() if v["label"].upper() == "REAL")
        fake = sorted(k for k, v in meta.items() if v["label"].upper() == "FAKE")
        n_real = min(len(real), n // 2)
        chosen = list(rng.choice(real, n_real, replace=False)) if n_real else []
        n_fake = min(len(fake), n - n_real)
        chosen += list(rng.choice(fake, n_fake, replace=False)) if n_fake else []
        members = {folder + name: name for name in chosen}
        safe_extract(zf, out / "videos", list(members), lambda m: members[m])
    subset = {k: meta[k] for k in sorted(chosen)}
    (out / "metadata.json").write_text(json.dumps(subset, indent=1))
    return {"real": n_real, "fake": n_fake}


def arrange_folder(dataset: str, out: Path, real: Sequence[Path], fake: Sequence[Path],
                   method: str) -> None:  # fmt: skip
    """Link (or copy, if links are not allowed) source folders into the real/ fake/<method>/
    layout that the folder indexer reads. Nothing is moved or deleted."""
    for src, dst in [*((r, out / "real" / r.name) for r in real),
                     *((f, out / "fake" / method / f.name) for f in fake)]:  # fmt: skip
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        except OSError:  # Windows without developer mode cannot make links
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)


# ----------------------------------------------------------------------------- checks


@dataclass
class Check:
    dataset: str
    ok: bool
    detail: str


EXPECTED_FFPP = {"real": 1000, **{m: 1000 for m in FFPP_METHODS}}


def check_ffpp(root: Path, compression: str) -> Check:
    if not (root / "splits" / "train.json").exists():
        return Check(f"ffpp {compression}", False, "split lists missing (run the ffpp step)")
    counts = {
        "real": len(
            list((root / "original_sequences/youtube" / compression / "videos").glob("*.mp4"))
        )
    }
    for m in FFPP_METHODS:
        counts[m] = len(
            list((root / "manipulated_sequences" / m / compression / "videos").glob("*.mp4"))
        )
    sizes = {
        s: len(json.loads((root / "splits" / f"{s}.json").read_text())) * 2
        for s in ("train", "val", "test")
    }
    ok = counts == EXPECTED_FFPP and sizes == {"train": 720, "val": 140, "test": 140}
    detail = ", ".join(f"{k} {v}/{EXPECTED_FFPP[k]}" for k, v in counts.items())
    return Check(f"ffpp {compression}", ok, f"{detail}; splits {sizes}")


def check_celebdf(root: Path) -> Check:
    if not (root / "List_of_testing_videos.txt").exists():
        return Check("celebdf_v2", False, "List_of_testing_videos.txt missing")
    recs = INDEXERS["celebdf_v2"](root)
    real = sum(r.label == 0 for r in recs)
    ok = len(recs) == 518
    return Check(
        "celebdf_v2", ok, f"{len(recs)}/518 test videos ({real} real, {len(recs) - real} fake)"
    )


def check_indexed(dataset: str, root: Path) -> Check:
    if not root.exists():
        return Check(dataset, False, "not downloaded")
    recs = INDEXERS[dataset](root)
    real = sum(r.label == 0 for r in recs)
    missing = sum(not r.path.exists() for r in recs)
    ok = len(recs) > 0 and real > 0 and real < len(recs) and missing == 0
    detail = f"{len(recs)} items ({real} real, {len(recs) - real} fake)"
    return Check(dataset, ok, detail + (f", {missing} missing files" if missing else ""))


def check_all(raw: Path) -> list[Check]:
    out = []
    ffpp = raw / "ffpp"
    for comp in ("c23", "c40"):
        out.append(
            check_ffpp(ffpp, comp)
            if ffpp.exists()
            else Check(f"ffpp {comp}", False, "not downloaded")
        )
    celeb = raw / "celebdf_v2"
    out.append(
        check_celebdf(celeb) if celeb.exists() else Check("celebdf_v2", False, "not downloaded")
    )
    for d in ("dfdc", "deeperforensics", "wilddeepfake", "df40"):
        out.append(check_indexed(d, raw / d))
    return out


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("disk", help="free space against the plan's needs")
    f = sub.add_parser("ffpp", help="run the official FF++ script you received by email")
    f.add_argument("--script", type=Path, required=True)
    f.add_argument("--compression", nargs="+", default=["c23", "c40"])
    f.add_argument("--server", default="EU2", choices=["EU", "EU2", "CA"])
    c = sub.add_parser("celebdf", help="unpack the Celeb-DF v2 zip")
    c.add_argument("--zip", type=Path, required=True)
    d = sub.add_parser("dfdc", help="take a fixed subset from one DFDC part zip")
    d.add_argument("--zip", type=Path, required=True)
    d.add_argument("--n", type=int, default=1000)
    d.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("folder", help="arrange DeeperForensics, WildDeepfake or DF40")
    g.add_argument("--dataset", required=True, choices=["deeperforensics", "wilddeepfake", "df40"])
    g.add_argument("--real", type=Path, nargs="+", required=True)
    g.add_argument("--fake", type=Path, nargs="+", required=True)
    g.add_argument("--method", required=True, help="name of the fake-making method")
    sub.add_parser("check", help="count files in every dataset")
    args = p.parse_args(argv)

    if args.cmd == "disk":
        free = free_gb(args.raw)
        print(f"Free: {free:.1f} GB. Plan needs about {NEEDED_GB} GB; rough sizes: {ROUGH_GB}")
        if free < NEEDED_GB:
            print("Low on space: use c23 only (c40 test videos only), the DFDC subset, and delete "
                  "raw videos after their crops pass the check.")  # fmt: skip
    elif args.cmd == "ffpp":
        get_ffpp(args.script, args.raw / "ffpp", args.compression, args.server)
    elif args.cmd == "celebdf":
        print(f"extracted {get_celebdf(args.zip, args.raw / 'celebdf_v2')} files")
    elif args.cmd == "dfdc":
        print(f"DFDC subset: {get_dfdc_subset(args.zip, args.raw / 'dfdc', args.n, args.seed)}")
    elif args.cmd == "folder":
        arrange_folder(args.dataset, args.raw / args.dataset, args.real, args.fake, args.method)
    if args.cmd in ("ffpp", "celebdf", "dfdc", "folder", "check"):
        checks = check_all(args.raw)
        for ch in checks:
            print(f"{'OK  ' if ch.ok else '--  '} {ch.dataset:16} {ch.detail}")
        return 0 if all(ch.ok for ch in checks if ch.detail != "not downloaded") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
