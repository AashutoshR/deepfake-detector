import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deepfake_detector import experiments
from deepfake_detector.experiments import (
    EXPERIMENTS,
    TARGETS,
    estimate_hours,
    gather,
    plan,
    results_markdown,
    verdicts,
)

SETS = ["ffpp_c23", "ffpp_c40", "celebdf_v2", "dfdc"]
# Per experiment: (video AUC on every set, held-out AUC, ECE)
QUALITY = {
    "final": (0.90, 0.88, 0.06),
    "plain_vit": (0.75, 0.70, 0.20),
    "xception": (0.78, 0.86, 0.15),
    "efficientnet_b4": (0.80, 0.80, 0.12),
}


def fake_runner(calls: list[list[str]]):
    def run(args: list[str]) -> None:
        calls.append(args)
        if args[0] != "deepfake_detector.evaluate":
            return
        out = Path(args[args.index("--out") + 1])
        name = out.name
        exp = next(e for e in QUALITY if name.startswith(f"exp_{e}_s"))
        seed = int(name.split("_s")[-1].split("__")[0])
        auc, held, ece = QUALITY[exp]
        jitter = 0.01 * seed
        sets = {}
        for s in args[args.index("--sets") + 1 :]:
            sets[s] = {"status": "ok", "video_auc": auc + jitter, "video_ece": ece}
            if "__lomo-" in name:
                method = name.split("__lomo-")[1]
                sets[s]["per_method_auc"] = {method: held + jitter}
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps({"sets": sets}))

    return run


def test_plan_and_estimate() -> None:
    runs = plan(None, [0, 1, 2], ["Deepfakes", "FaceSwap"], SETS)
    assert len(runs) == 4 * 3 * 3
    assert runs["exp_final_s1"].overrides == ["train.seed=1"]
    assert runs["exp_final_s1"].sets == SETS
    lomo = runs["exp_xception_s2__lomo-FaceSwap"]
    assert lomo.overrides == [
        "model.arch=xception",
        "train.seed=2",
        "train.leave_out_method=FaceSwap",
    ]
    assert lomo.sets == ["ffpp_c23"]
    epochs = dict.fromkeys(EXPERIMENTS, 20)
    one = {"exp_final_s0": runs["exp_final_s0"]}
    assert estimate_hours(one, epochs) == pytest.approx(3.5)
    slow = {"r": runs["exp_efficientnet_b4_s0"]}
    assert estimate_hours(slow, epochs) > 8  # EfficientNet-B4 trains about 2.5x slower


def test_main_runs_calibrates_and_reports(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    out = tmp_path / "out"
    eval_cfg = tmp_path / "eval.yaml"
    cfg = yaml.safe_load(Path("configs/eval.yaml").read_text())
    cfg["test_sets"] = SETS
    eval_cfg.write_text(yaml.safe_dump(cfg))
    df = experiments.main(
        [
            "--seeds",
            "0",
            "1",
            "--m5-methods",
            "Deepfakes",
            "--epochs",
            "3",
            "--baseline-epochs",
            "2",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out",
            str(out),
            "--eval-config",
            str(eval_cfg),
        ],
        runner=fake_runner(calls),
    )
    trains = [c for c in calls if c[0] == "deepfake_detector.train"]
    assert any("train.epochs=3" in c and "train.run_name=exp_final_s0" in c for c in trains)
    assert any("train.epochs=2" in c and "model.arch=xception" in c for c in trains)
    assert sum(c[0] == "deepfake_detector.calibrate" for c in calls) == len(trains)

    final_m2 = df[(df.experiment == "final") & (df.target == "M2")].iloc[0]
    assert final_m2["mean"] == pytest.approx(0.905) and final_m2["seeds_done"] == 2
    assert final_m2["std"] == pytest.approx(np.std([0.90, 0.91], ddof=1))
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "0.905 ± 0.007" in md
    assert "- M2 vs xception: yes" in md
    assert "- M2: stretch met" in md and "- M7: met" in md
    assert "- M3: stretch met" in md  # 0.905 clears both 0.70 and the 0.75 stretch
    assert "- M4a: not run" in md  # DeeperForensics not in this run's test sets

    calls.clear()
    experiments.main(
        [
            "--seeds",
            "0",
            "1",
            "--m5-methods",
            "Deepfakes",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out",
            str(out),
            "--eval-config",
            str(eval_cfg),
        ],
        runner=fake_runner(calls),
    )
    assert calls == []  # all done: nothing re-run


def test_verdicts_detect_a_loss_and_missing_seeds(tmp_path: Path) -> None:
    out = tmp_path
    run = fake_runner([])
    for exp in ("final", "xception"):
        for seed in (0,):
            run(
                [
                    "deepfake_detector.evaluate",
                    "--out",
                    str(out / f"exp_{exp}_s{seed}"),
                    "--sets",
                    "celebdf_v2",
                    "ffpp_c23",
                ]
            )
            run(
                [
                    "deepfake_detector.evaluate",
                    "--out",
                    str(out / f"exp_{exp}_s{seed}__lomo-Deepfakes"),
                    "--sets",
                    "ffpp_c23",
                ]
            )
    df = gather(out, ["final", "xception"], [0, 1], ["Deepfakes"])
    v = verdicts(df, n_seeds=2)
    assert v["targets"]["M2"] == "stretch met (incomplete seeds)"
    assert v["beats_baselines"]["M5 vs xception"] is True  # 0.88 > 0.86
    assert v["targets"]["M6"] == "not run"
    md = results_markdown(df, v, [0, 1])
    assert "(1/2 seeds)" in md
    assert set(TARGETS) >= {"M1", "M2", "M5", "M7"}


def test_dry_run_lists_runs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    experiments.main(
        [
            "--dry-run",
            "--only",
            "final",
            "--seeds",
            "0",
            "--m5-methods",
            "Deepfakes",
            "--out",
            str(tmp_path),
        ]
    )
    printed = capsys.readouterr().out
    assert "exp_final_s0 " in printed and "exp_final_s0__lomo-Deepfakes" in printed
