import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from api.server import create_app
from fastapi.testclient import TestClient

from deepfake_detector import teach as teach_mod
from deepfake_detector.predict import Predictor
from deepfake_detector.teach import example_counts, mix, save_example


def test_save_example_dedupes_and_relabels(tmp_path: Path) -> None:
    a = save_example(b"one", ".png", "fake", tmp_path)
    assert a.parent == tmp_path / "fake" / "user" and a.read_bytes() == b"one"
    assert save_example(b"one", ".PNG", "fake", tmp_path) == a  # same content, kept once
    b = save_example(b"one", ".png", "real", tmp_path)  # marked again the other way: moved
    assert not a.exists() and b.exists()
    save_example(b"two", ".mp4", "fake", tmp_path)
    assert example_counts(tmp_path) == {"real": 1, "fake": 1}
    assert example_counts(tmp_path / "nothing") == {"real": 0, "fake": 0}
    with pytest.raises(ValueError):
        save_example(b"x", ".png", "maybe", tmp_path)
    with pytest.raises(ValueError):
        save_example(b"x", "/../x", "fake", tmp_path)


def test_mix_repeats_examples_and_balances_replay() -> None:
    ex = pd.DataFrame({"video_id": ["e"], "label": [1]})
    old = pd.DataFrame({"video_id": [f"o{i}" for i in range(30)], "label": [0] * 20 + [1] * 10})
    out = mix(ex, old, repeats=5, n_replay=16, seed=0)
    assert (out["video_id"] == "e").sum() == 5
    assert out[out["video_id"] != "e"]["label"].value_counts().to_dict() == {0: 8, 1: 8}


def test_feedback_endpoint(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DFD_FEEDBACK_DIR", str(tmp_path))
    client = TestClient(create_app(Predictor(checkpoint, device="cpu")))
    r = client.post("/feedback", files={"file": ("x.png", b"img", "image/png")},
                    data={"label": "fake"})  # fmt: skip
    assert r.status_code == 200 and r.json()["counts"] == {"real": 0, "fake": 1}
    r = client.post("/feedback", files={"file": ("x.png", b"img", "image/png")},
                    data={"label": "unsure"})  # fmt: skip
    assert r.status_code == 422
    r = client.post("/feedback", files={"file": ("x.exe", b"img", "application/x")},
                    data={"label": "fake"})  # fmt: skip
    assert r.status_code == 415


def test_teach_learns_marked_examples(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mark validation reals as fakes: after teaching they should score higher."""
    cfg = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
    manifest = Path(cfg["data"]["manifest_dir"]) / cfg["train"]["dataset"] / "manifest.parquet"
    df = pd.read_parquet(manifest)
    marked = df[(df["split"] == "val") & (df["label"] == 0)].head(4).assign(label=1)
    monkeypatch.setattr(teach_mod, "prepare", lambda *a, **k: marked.reset_index(drop=True))
    save_example(b"pretend", ".png", "fake", tmp_path)

    report = teach_mod.teach(
        checkpoint, tmp_path, epochs=3, repeats=8, n_replay=16, lr=1e-3, max_drop=1.0
    )
    assert report["kept"] and report["examples"] == {"real": 0, "fake": 1}
    rows = report["per_example"]
    assert rows and all(r["label"] == 1 for r in rows)
    # raw logits: the before and after probabilities use different calibrations
    assert all(r["logit_after"] > r["logit_before"] for r in rows)
    out = checkpoint.parent.parent / f"{checkpoint.parent.name}_taught"
    assert (out / "best.pt").exists() and (out / "calibration.json").exists()
    assert json.loads((out / "teach.json").read_text())["kept"]
    note = Predictor(out / "best.pt", device="cpu").model_note
    assert note is not None and "Fine-tuned on 1 fake and 0 real" in note

    strict = teach_mod.teach(checkpoint, tmp_path, epochs=1, repeats=1, n_replay=4, max_drop=-1.0)
    assert not strict["kept"] and not (out / "best.pt").exists()


def test_index_folders_with_only_fakes(tmp_path: Path) -> None:
    from deepfake_detector.data.indexers import index_folders

    save_example(b"img", ".png", "fake", tmp_path)
    recs = index_folders(tmp_path, "feedback", split="train")
    assert [(r.label, r.method) for r in recs] == [(1, "user")]
