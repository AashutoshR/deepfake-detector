import json
from pathlib import Path

import pandas as pd
import pytest

from deepfake_detector.data.indexers import FAKE, REAL, index_celebdf, index_ffpp
from deepfake_detector.data.manifest import SplitLeakError, check_no_leaks


def touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


@pytest.fixture
def ffpp_root(tmp_path: Path) -> Path:
    root = tmp_path / "ffpp"
    splits = {"train": [["000", "003"]], "val": [["001", "004"]], "test": [["002", "005"]]}
    (root / "splits").mkdir(parents=True)
    for name, pairs in splits.items():
        (root / "splits" / f"{name}.json").write_text(json.dumps(pairs))
    for sid in ["000", "001", "002", "003", "004", "005"]:
        touch(root / "original_sequences/youtube/c23/videos" / f"{sid}.mp4")
    for method in ["Deepfakes", "FaceSwap"]:
        for a, b in [("000", "003"), ("003", "000"), ("002", "005")]:
            touch(root / f"manipulated_sequences/{method}/c23/videos/{a}_{b}.mp4")
    return root


def test_index_ffpp_splits_and_labels(ffpp_root: Path) -> None:
    recs = index_ffpp(ffpp_root, methods=["Deepfakes", "FaceSwap"])
    assert len(recs) == 6 + 6
    by_id = {r.video_id: r for r in recs}
    assert by_id["real/000"].split == "train" and by_id["real/000"].label == REAL
    fake = by_id["Deepfakes/002_005"]
    assert fake.split == "test" and fake.label == FAKE and fake.method == "Deepfakes"
    assert set(fake.persons) == {"ffpp:002", "ffpp:005"}


def test_index_ffpp_filters(ffpp_root: Path) -> None:
    recs = index_ffpp(ffpp_root, methods=["Deepfakes"], splits=["test"])
    assert {r.video_id for r in recs} == {"real/002", "real/005", "Deepfakes/002_005"}


def test_index_celebdf_labels_from_folder(tmp_path: Path) -> None:
    root = tmp_path / "celeb"
    for rel in [
        "Celeb-real/id0_0000.mp4",
        "Celeb-synthesis/id0_id1_0000.mp4",
        "YouTube-real/00001.mp4",
        "Celeb-real/id2_0001.mp4",
    ]:
        touch(root / rel)
    # In the official list 1 means real and 0 means fake; we ignore those numbers.
    (root / "List_of_testing_videos.txt").write_text(
        "1 Celeb-real/id0_0000.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n1 YouTube-real/00001.mp4\n"
    )
    recs = index_celebdf(root)
    by_id = {r.video_id: r for r in recs}
    assert set(by_id) == {
        "Celeb-real/id0_0000",
        "Celeb-synthesis/id0_id1_0000",
        "YouTube-real/00001",
    }
    assert by_id["Celeb-synthesis/id0_id1_0000"].label == FAKE
    assert by_id["Celeb-real/id0_0000"].label == REAL
    assert by_id["Celeb-synthesis/id0_id1_0000"].persons == ("celebdf:id0", "celebdf:id1")
    assert len(index_celebdf(root, only_test=False)) == 4


def manifest(rows: list[tuple[str, str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dataset": "ffpp", "video_id": v, "split": s, "persons": p} for v, s, p in rows]
    )


def test_check_no_leaks_passes_clean() -> None:
    check_no_leaks(
        manifest([("a", "train", ["p1"]), ("a", "train", ["p1"]), ("b", "test", ["p2"])])
    )


def test_check_no_leaks_video_in_two_splits() -> None:
    with pytest.raises(SplitLeakError, match="videos"):
        check_no_leaks(manifest([("a", "train", ["p1"]), ("a", "test", ["p1"])]))


def test_check_no_leaks_person_in_two_splits() -> None:
    with pytest.raises(SplitLeakError, match="people"):
        check_no_leaks(manifest([("a", "train", ["p1"]), ("b", "test", ["p1", "p2"])]))


def test_official_ffpp_split_has_no_leaks(ffpp_root: Path) -> None:
    recs = index_ffpp(ffpp_root)
    df = pd.DataFrame(
        [
            {
                "dataset": r.dataset,
                "video_id": r.video_id,
                "split": r.split,
                "persons": list(r.persons),
            }
            for r in recs
        ]
    )
    check_no_leaks(df)
