import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from deepfake_detector import ablations, evaluate
from deepfake_detector.ablations import (
    ABLATIONS,
    effective_overrides,
    execute,
    gather,
    plan,
    run_name,
    table_markdown,
)
from deepfake_detector.explain import CASES, _target_layer, grad_cam, heatmap_report, pick_examples
from deepfake_detector.models import build_model
from deepfake_detector.train import load_config, train
from tests.test_train import build_dataset, tiny_cfg

# ----------------------------------------------------------------------------- Grad-CAM


@pytest.mark.parametrize(
    "cfg",
    [
        {"arch": "vit", "pretrained": "none", "timm_name": "vit_tiny_patch16_224", "img_size": 64},
        {"arch": "efficientnet_b4"},
        {"arch": "xception"},
    ],
)
def test_grad_cam_every_arch(cfg: dict) -> None:
    model = build_model(cfg, pretrained_weights=False)
    heat = grad_cam(model, torch.rand(2, 3, 64, 64))
    assert heat.shape == (2, 64, 64)
    assert heat.min() >= 0 and heat.max() <= 1
    assert model.training


def test_target_layer_unknown() -> None:
    with pytest.raises(ValueError):
        _target_layer(torch.nn.Linear(2, 2))


def test_pick_examples_includes_failures_one_per_video() -> None:
    frames = pd.DataFrame(
        {
            "video_id": ["f1", "f1", "f2", "r1", "r2", "r2"],
            "label": [1, 1, 1, 0, 0, 0],
            "method": ["Deepfakes"] * 3 + ["real"] * 3,
            "prob": [0.9, 0.95, 0.2, 0.1, 0.8, 0.7],
        }
    )
    ex = pick_examples(frames, n=2)
    assert set(ex["case"]) == set(CASES)
    caught = ex[ex["case"] == "fake, caught"]
    assert caught["video_id"].tolist() == ["f1"] and caught["prob"].iloc[0] == 0.95
    assert ex[ex["case"] == "real, flagged (false alarm)"]["prob"].iloc[0] == 0.8
    assert ex[ex["case"] == "fake, missed"]["video_id"].tolist() == ["f2"]


def test_heatmap_report_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "data"
    build_dataset(root)
    run = Path(train(tiny_cfg(root, "train.epochs=1"), pretrained_weights=False)["run_dir"])
    eval_cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())
    eval_cfg.update(num_workers=0, batch_size=8)
    eval_cfg["bootstrap"]["n"] = 10
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump(eval_cfg))
    out = tmp_path / "eval"
    evaluate.main(
        [
            "--checkpoint",
            str(run / "best.pt"),
            "--config",
            str(tmp_path / "eval.yaml"),
            "--out",
            str(out),
            "--sets",
            "ffpp_c23",
            "celebdf_v2",
        ]
    )
    frames = pd.read_parquet(out / "ffpp_c23_frames.parquet")
    assert {"crop_path", "frame"} <= set(frames.columns)

    written = heatmap_report(run / "best.pt", out, n=2)
    assert [p.name for p in written] == ["heatmaps_ffpp_c23.png"]
    md = (out / "heatmaps.md").read_text()
    assert "ffpp_c23" in md and "celebdf_v2" not in md


# ----------------------------------------------------------------------------- ablations


@pytest.fixture(scope="module")
def base():
    return load_config()


def test_effective_overrides_drop_defaults(base) -> None:
    assert effective_overrides(base, ["model.pretrained=imagenet21k"]) == []
    assert effective_overrides(base, ["model.pretrained=clip"]) == ["model.pretrained=clip"]
    assert effective_overrides(base, ["train.compressions=[c23]"]) == []
    assert effective_overrides(base, ["train.compressions=[c23,c40]"]) == [
        "train.compressions=[c23,c40]"
    ]


def test_run_names_are_distinct() -> None:
    assert run_name([]) == "abl_base"
    assert run_name([], "Deepfakes") == "abl_base__lomo-Deepfakes"
    assert run_name(["train.sbi.enabled=false"]) != run_name(["model.method_head.enabled=false"])


def test_plan_shares_base_runs(base) -> None:
    runs = plan(["A1", "A2", "A8"], m5_methods=["NeuralTextures"], base=base)
    # A1: 3 options (ImageNet-21k is the base), A2: 3 (LoRA is the base), A8: base only.
    # Distinct trained settings: base, clip, dinov2, full, last layer = 5, each with 1 M5 run.
    assert len(runs) == 10
    users = runs["abl_base"].used_by
    assert "A1 ImageNet-21k" in users and "A2 LoRA" in users and "A8 Median" in users
    assert runs["abl_base__lomo-NeuralTextures"].sets == ["ffpp_c23"]
    assert runs["abl_base"].sets == ["ffpp_c23", "celebdf_v2"]


def test_full_grid_size(base) -> None:
    runs = plan(base=base)
    trained = {n.split("__")[0] for n in runs}
    assert len(trained) == 11 and len(runs) == 55
    assert set(ABLATIONS) == {f"A{i}" for i in range(1, 9)}


def fake_runner(out_root: Path, calls: list[list[str]]):
    """Stands in for training and testing: writes results a real run would produce."""

    def run(args: list[str]) -> None:
        calls.append(args)
        if args[0] != "deepfake_detector.evaluate":
            return
        out = Path(args[args.index("--out") + 1])
        name = out.name
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(len(name))
        sets = {}
        for s in args[args.index("--sets") + 1 :]:
            frames = pd.DataFrame(
                {
                    "video_id": np.repeat([f"v{i}" for i in range(8)], 3),
                    "label": np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3),
                    "method": np.repeat(["real"] * 4 + ["NeuralTextures"] * 4, 3),
                    "prob": rng.random(24),
                }
            )
            frames.to_parquet(out / f"{s}_frames.parquet")
            sets[s] = {
                "status": "ok",
                "video_auc": 0.5 + len(name) % 7 / 20,
                "per_method_auc": {"NeuralTextures": 0.7},
            }
        (out / "results.json").write_text(json.dumps({"sets": sets}))

    return run


def test_execute_resumes_and_skips(tmp_path: Path, base) -> None:
    runs = plan(["A5"], m5_methods=["NeuralTextures"], base=base)
    runs_dir, out = tmp_path / "runs", tmp_path / "out"
    (runs_dir / "abl_base").mkdir(parents=True)
    (runs_dir / "abl_base" / "last.pt").write_bytes(b"")
    calls: list[list[str]] = []
    execute(
        runs, runs_dir, out, tmp_path / "eval.yaml", ["train.epochs=2"], fake_runner(out, calls)
    )
    trains = [c for c in calls if c[0] == "deepfake_detector.train"]
    assert ["--resume", str(runs_dir / "abl_base" / "last.pt")] == trains[0][1:]
    assert any("model.pooling=mean_patch" in c and "train.epochs=2" in c for c in trains)
    n_calls = len(calls)
    execute(runs, runs_dir, out, tmp_path / "eval.yaml", [], fake_runner(out, calls))
    assert len(calls) == n_calls  # everything done: nothing re-run


def test_main_gathers_table_including_a8(tmp_path: Path, base) -> None:
    out = tmp_path / "out"
    calls: list[list[str]] = []
    df = ablations.main(
        [
            "--only",
            "A5",
            "A8",
            "--m5-methods",
            "NeuralTextures",
            "--epochs",
            "2",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out",
            str(out),
        ],
        runner=fake_runner(out, calls),
    )
    assert len(df) == 5 and not df[["M1", "M2", "M5"]].isna().any().any()
    a8 = df[df["ablation"] == "A8"]
    assert a8["M1"].nunique() > 1  # mean, median and top-k really re-score
    md = (out / "ablations.md").read_text()
    assert "| A5 | Token pooling | Mean of patch tokens |" in md and "**" in md
    assert (out / "ablations.json").exists() and (out / "eval.yaml").exists()


def test_gather_marks_missing_runs(tmp_path: Path, base) -> None:
    df = gather(tmp_path, ["A4"], ["Deepfakes"], base)
    assert df[["M1", "M2", "M5"]].isna().all().all()
    assert set(df["M5_runs_done"]) == {"0/1"}
    assert "n/a" in table_markdown(df, 8, ["Deepfakes"])


def test_dry_run_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ablations.main(
        ["--dry-run", "--only", "A4", "--m5-methods", "Deepfakes", "--out", str(tmp_path)]
    )
    printed = capsys.readouterr().out
    assert "abl_base" in printed and "abl_augment.preset-basic" in printed
