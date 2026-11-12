import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_detector.data import download
from deepfake_detector.data.download import (
    arrange_folder,
    check_all,
    check_ffpp,
    free_gb,
    get_celebdf,
    get_dfdc_subset,
    get_ffpp,
    safe_extract,
)
from deepfake_detector.data.extract import frame_files, read_media
from deepfake_detector.data.indexers import FAKE, REAL, index_dfdc, index_folders
from tests.conftest import write_video


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def test_safe_extract_blocks_zip_slip(tmp_path: Path) -> None:
    for bad in ["../evil.txt", "a/../../evil.txt", "/abs.txt", "C:/win.txt"]:
        z = make_zip(tmp_path / "bad.zip", {bad: b"x"})
        with zipfile.ZipFile(z) as zf, pytest.raises(ValueError):
            safe_extract(zf, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_get_celebdf_strips_top_folder(tmp_path: Path) -> None:
    z = make_zip(
        tmp_path / "c.zip",
        {
            "Celeb-DF-v2/Celeb-real/id0_0000.mp4": b"v",
            "Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4": b"v",
            "Celeb-DF-v2/List_of_testing_videos.txt": b"1 Celeb-real/id0_0000.mp4\n",
        },
    )
    assert get_celebdf(z, tmp_path / "celeb") == 3
    assert (tmp_path / "celeb" / "Celeb-real" / "id0_0000.mp4").exists()
    # Already-flat archives are kept as they are.
    z2 = make_zip(tmp_path / "c2.zip", {"Celeb-real/id1_0000.mp4": b"v"})
    get_celebdf(z2, tmp_path / "celeb2")
    assert (tmp_path / "celeb2" / "Celeb-real" / "id1_0000.mp4").exists()


def test_dfdc_subset_is_fixed_and_balanced(tmp_path: Path) -> None:
    meta = {f"r{i}.mp4": {"label": "REAL"} for i in range(5)}
    meta |= {f"f{i}.mp4": {"label": "FAKE", "original": f"r{i % 5}.mp4"} for i in range(20)}
    files = {f"dfdc_train_part_0/{k}": b"v" for k in meta}
    files["dfdc_train_part_0/metadata.json"] = json.dumps(meta).encode()
    z = make_zip(tmp_path / "d.zip", files)

    counts = get_dfdc_subset(z, tmp_path / "a", n=10, seed=0)
    assert counts == {"real": 5, "fake": 5}
    chosen = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert len(list((tmp_path / "a" / "videos").glob("*.mp4"))) == 10
    get_dfdc_subset(z, tmp_path / "b", n=10, seed=0)
    assert chosen == json.loads((tmp_path / "b" / "metadata.json").read_text())

    recs = index_dfdc(tmp_path / "a")
    assert sum(r.label == REAL for r in recs) == 5
    fake = next(r for r in recs if r.label == FAKE)
    assert fake.persons[0].startswith("dfdc:r") and fake.split == "test"


def test_dfdc_needs_one_metadata(tmp_path: Path) -> None:
    z = make_zip(tmp_path / "d.zip", {"x.mp4": b"v"})
    with pytest.raises(ValueError):
        get_dfdc_subset(z, tmp_path / "out")


def write_frames(folder: Path, n: int) -> None:
    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i}.png"), np.full((40, 40, 3), i * 10, np.uint8))


def test_frame_folders_read_in_natural_order(tmp_path: Path) -> None:
    write_frames(tmp_path / "seq", 12)
    assert [p.name for p in frame_files(tmp_path / "seq")][:3] == ["0.png", "1.png", "2.png"]
    frames = read_media(tmp_path / "seq", 4)
    assert [i for i, _ in frames] == [1, 4, 7, 10]
    assert frames[1][1].mean() == pytest.approx(40, abs=1)


def test_index_folders_videos_images_and_sequences(tmp_path: Path) -> None:
    root = tmp_path / "wild"
    write_video(root / "real" / "a.mp4", 5)
    write_frames(root / "real" / "seq1", 3)
    write_frames(root / "fake" / "wild" / "seq2", 3)
    (root / "fake" / "gan").mkdir(parents=True)  # cv2.imwrite fails silently without it
    assert cv2.imwrite(str(root / "fake" / "gan" / "img.png"), np.zeros((8, 8, 3), np.uint8))
    recs = index_folders(root, "wilddeepfake")
    by_id = {r.video_id: r for r in recs}
    assert set(by_id) == {"real/a", "real/seq1", "fake/wild/seq2", "fake/gan/img"}
    assert by_id["fake/wild/seq2"].method == "wild" and by_id["fake/wild/seq2"].label == FAKE
    assert by_id["real/seq1"].path.is_dir()


def test_arrange_folder_and_check(tmp_path: Path) -> None:
    src_real, src_fake = tmp_path / "src_real", tmp_path / "src_fake"
    write_video(src_real / "r.mp4", 4)
    write_video(src_fake / "f.mp4", 4)
    raw = tmp_path / "raw"
    arrange_folder("deeperforensics", raw / "deeperforensics", [src_real], [src_fake], "e2e")
    assert (raw / "deeperforensics" / "fake" / "e2e" / "src_fake").exists()
    arrange_folder("deeperforensics", raw / "deeperforensics", [src_real], [src_fake], "e2e")
    checks = {c.dataset: c for c in check_all(raw)}
    assert checks["deeperforensics"].ok
    assert checks["deeperforensics"].detail.startswith("2 items (1 real, 1 fake)")
    assert checks["celebdf_v2"].detail == "not downloaded" and not checks["ffpp c23"].ok


def test_get_ffpp_runs_official_script_and_fetches_splits(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    pairs = {
        "train": [["000", "001"]] * 360,
        "val": [["002", "003"]] * 70,
        "test": [["004", "005"]] * 70,
    }

    def fetch(url: str) -> bytes:
        return json.dumps(pairs[url.rsplit("/", 1)[1].removesuffix(".json")]).encode()

    out = tmp_path / "ffpp"
    get_ffpp(Path("dl.py"), out, ["c23"], "EU2", run=calls.append, fetch=fetch)
    assert [c[4] for c in calls] == [
        "original",
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    ]
    assert calls[0][-4:] == ["-t", "videos", "--server", "EU2"]
    check = check_ffpp(out, "c23")
    assert not check.ok and "splits {'train': 720, 'val': 140, 'test': 140}" in check.detail
    assert "real 0/1000" in check.detail


def test_cli_disk_and_check(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert free_gb(tmp_path / "does" / "not" / "exist") > 0
    assert download.main(["--raw", str(tmp_path), "disk"]) == 0
    assert "Free:" in capsys.readouterr().out
    assert download.main(["--raw", str(tmp_path), "check"]) == 0  # nothing downloaded yet
    assert "not downloaded" in capsys.readouterr().out
