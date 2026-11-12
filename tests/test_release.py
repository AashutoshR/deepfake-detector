import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from deepfake_detector import calibrate, evaluate, release
from deepfake_detector.inference import load_detector
from deepfake_detector.predict import Predictor
from deepfake_detector.release import fill_block, fill_card, push, stage, validate
from deepfake_detector.train import train
from tests.conftest import FixedDetector, write_video
from tests.test_train import build_dataset, tiny_cfg


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """A tiny trained, calibrated and evaluated run: (checkpoint, results.json, tmp dir)."""
    tmp = tmp_path_factory.mktemp("rel")
    root = tmp / "data"
    build_dataset(root)
    run = Path(train(tiny_cfg(root, "train.epochs=1"), pretrained_weights=False)["run_dir"])
    calibrate.main(["--checkpoint", str(run / "best.pt")])
    cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())
    cfg.update(num_workers=0, batch_size=8)
    cfg["bootstrap"]["n"] = 10
    cfg["robustness"]["sweeps"] = {"jpeg": [50]}
    (tmp / "eval.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp / "eval"
    evaluate.main(
        [
            "--checkpoint",
            str(run / "best.pt"),
            "--config",
            str(tmp / "eval.yaml"),
            "--out",
            str(out),
            "--sets",
            "ffpp_c23",
            "celebdf_v2",
            "--robustness",
        ]
    )
    return run / "best.pt", out / "results.json", tmp


def test_stage_writes_hashed_release(evaluated: tuple[Path, Path, Path]) -> None:
    ckpt, results, tmp = evaluated
    out = tmp / "release"
    problems = stage(ckpt, results, out)
    for f in [
        "model.safetensors",
        "config.json",
        "calibration.json",
        "LICENSE",
        "README.md",
        "SHA256SUMS",
    ]:
        assert (out / f).exists(), f
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "Filled by scripts/release.py" not in card
    assert "| ffpp_c23 | same dataset | 4 |" in card
    assert "| celebdf_v2 | cross-dataset | not tested |" in card
    assert "| jpeg | 50 |" in card and "Temperature" in card
    digest = (out / "SHA256SUMS").read_text().split()[0]
    assert digest in card
    # The card still has hand-written sections to finish, and the headline result is missing.
    assert "unfinished section: Fairness (S-5)" in problems
    assert "unfinished section: Failure cases" in problems
    assert any("headline result missing" in p for p in problems)
    assert any("M5" in p for p in problems)
    assert "license: cc-by-nc-4.0" in card


def test_stage_needs_calibration(evaluated: tuple[Path, Path, Path], tmp_path: Path) -> None:
    ckpt, results, _ = evaluated
    lonely = tmp_path / "best.pt"
    lonely.write_bytes(ckpt.read_bytes())
    with pytest.raises(FileNotFoundError, match="calibrate"):
        stage(lonely, results, tmp_path / "out")


def test_released_weights_match_checkpoint(evaluated: tuple[Path, Path, Path]) -> None:
    ckpt, results, tmp = evaluated
    out = tmp / "release_load"
    m5 = tmp / "m5.json"
    m5.write_text(json.dumps({"Deepfakes": 0.8, "FaceSwap": 0.7}))
    problems = stage(ckpt, results, out, m5_path=m5)
    assert not any("M5" in p for p in problems)
    assert "Mean unseen-method AUC (M5): 0.750" in (out / "README.md").read_text(encoding="utf-8")

    a, _ = load_detector(ckpt, torch.device("cpu"))
    b, cfg = load_detector(out / "model.safetensors", torch.device("cpu"))
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(a(x)["logit"], b(x)["logit"])
    assert cfg["format"] == "deepfake-detector/1"

    video = tmp / "v.mp4"
    write_video(video, 10)
    p = Predictor(out / "model.safetensors", device="cpu", detector=FixedDetector(), frames=4)
    assert p.calibrated and p.predict(video).faces_found == 4


def complete_results() -> dict[str, Any]:
    return {"sets": {"celebdf_v2": {"status": "ok"}}}


def test_validate_passes_complete_card() -> None:
    card = "## Fairness (S-5)\nAll groups within 2 AUC points.\n"
    assert validate(card, complete_results(), {"Deepfakes": 0.9}) == []
    assert validate("## Fairness\n_To be filled: x._\n", complete_results(), {"a": 1.0}) == [
        "unfinished section: Fairness"
    ]


def test_fill_block_requires_markers() -> None:
    with pytest.raises(ValueError):
        fill_block("no markers here", "results", "x")
    with pytest.raises(ValueError):
        fill_card("", {"sets": {}}, {}, {})


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, repo_id: str, **kw: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw: Any) -> Any:
        self.calls.append(("upload_folder", kw))
        return type("Info", (), {"commit_url": "https://huggingface.co/x/commit/1"})()


def test_push_refuses_public_draft_but_allows_private(tmp_path: Path) -> None:
    api = FakeApi()
    with pytest.raises(RuntimeError, match="not ready"):
        push(tmp_path, "me/model", ["unfinished section: Fairness"], api=api)
    assert api.calls == []
    url = push(tmp_path, "me/model", ["unfinished section: Fairness"], private=True, api=api)
    assert url.endswith("/commit/1")
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder"]  # one upload, together
    assert api.calls[0][1]["private"] is True


def test_main_confirms_before_publishing(
    evaluated: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, results, tmp = evaluated
    pushed: list[str] = []
    monkeypatch.setattr(release, "push", lambda out, repo, problems, private: pushed.append(repo))
    base = ["--checkpoint", str(ckpt), "--results", str(results), "--out", str(tmp / "r")]

    assert release.main(base) == 1  # staged, but not ready
    assert release.main([*base, "--push", "--repo-id", "me/m"]) == 1  # public draft refused
    assert (
        release.main([*base, "--push", "--repo-id", "me/m", "--private"], confirm=lambda _: "wrong")
        == 1
    )
    assert pushed == []
    assert (
        release.main([*base, "--push", "--repo-id", "me/m", "--private"], confirm=lambda _: "me/m")
        == 0
    )
    assert pushed == ["me/m"]
