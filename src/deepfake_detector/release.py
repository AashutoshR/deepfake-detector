"""Stage and publish the weights, model card and hashes together (S-1, S-7, S-8).

Staging writes a release folder:
  model.safetensors   weights only (no optimizer state, no pickled code)
  config.json         what is needed to rebuild the model and crop faces the same way
  calibration.json    temperature and band cut-offs fitted on FF++ validation (required, S-3)
  README.md           the model card, with results, calibration, robustness and hashes filled
  LICENSE             CC BY-NC 4.0 notice for the weights
  SHA256SUMS          one line per file

Publishing uploads that folder to Hugging Face Hub in one commit, so the card never lags the
weights. It refuses while the card still has unfinished sections or the headline result is
missing, and asks you to type the repo id before anything is published.

    python scripts/release.py --checkpoint runs/<run>/best.pt \\
        --results outputs/eval/<run>/best/results.json --out release
    python scripts/release.py ... --push --repo-id <user>/deepfake-detector
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

FORMAT = "deepfake-detector/1"
LICENSE_TEXT = """The model weights in this repository are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/

They were trained on FaceForensics++, which allows research and non-commercial use only. Using
the weights also means respecting the terms of the datasets they were trained and tested on.

The source code (https://github.com/AashutoshR/deepfake-detector) is licensed separately under
the MIT License.
"""
PLACEHOLDER = re.compile(r"_(To be filled|Filled by scripts/release\.py)[^_]*_")
HEADLINE_SET = "celebdf_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_weights(checkpoint: Path, out: Path) -> dict[str, Any]:
    """Write model.safetensors and config.json; return the config."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in ckpt["model"].items()}
    save_file(state, str(out / "model.safetensors"), metadata={"format": FORMAT})
    cfg = ckpt["config"]
    t = cfg["train"]
    config = {
        "format": FORMAT,
        "model": cfg["model"],
        "data": {"faces": cfg["data"]["faces"]},
        "train": {
            k: t.get(k) for k in ("dataset", "compressions", "leave_out_method", "seed", "epochs")
        },  # fmt: skip
        "trained_epochs": ckpt.get("epoch"),
        "val_metrics": ckpt.get("metrics", {}),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return config


# ----------------------------------------------------------------------------- card


def _f(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def _results_table(results: dict[str, Any]) -> str:
    lines = [
        "| Test set | Kind | Videos | Video AUC (95% CI) | Frame AUC | EER | Accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results["sets"].items():
        kind = "same dataset" if name.startswith("ffpp") else "cross-dataset"
        if m.get("status") != "ok":
            lines.append(f"| {name} | {kind} | not tested | | | | |")
            continue
        lo, hi = m["video_auc_ci"]
        lines.append(
            f"| {name} | {kind} | {int(m['n_videos'])} | {_f(m['video_auc'])} ({_f(lo)} to "
            f"{_f(hi)}) | {_f(m['frame_auc'])} | {_f(m['video_eer'])} | {_f(m['video_acc'])} |"
        )
    return "\n".join(lines)


def _per_method(results: dict[str, Any], m5: dict[str, float] | None) -> str:
    per = results["sets"].get("ffpp_c23", {}).get("per_method_auc", {})
    lines = [
        "Video AUC per FF++ method (c23 test), and on each method when it was left out of "
        "training (M5, unseen method):",
        "",
        "| Method | Seen in training | Left out of training (M5) |",
        "|---|---|---|",
    ]
    held = m5 or {}
    for method in sorted(set(per) | set(held)):
        lines.append(f"| {method} | {_f(per.get(method))} | {_f(held.get(method))} |")
    if m5:
        lines += ["", f"Mean unseen-method AUC (M5): {_f(sum(m5.values()) / len(m5))}"]
    return "\n".join(lines)


def _robustness(results: dict[str, Any]) -> str:
    rob = results.get("robustness")
    if not rob or "sweeps" not in rob:
        return "_To be filled: run evaluate with --robustness._"
    lines = [
        f"Video AUC on {rob['test_set']} (clean: {_f(rob['clean'])}) as each degradation "
        "gets stronger:",
        "",
        "| Degradation | Strength | Video AUC |",
        "|---|---|---|",
    ]
    for kind, vals in rob["sweeps"].items():
        lines += [f"| {kind} | {s} | {_f(v)} |" for s, v in vals.items()]
    return "\n".join(lines)


def _calibration(results: dict[str, Any], cal: dict[str, Any]) -> str:
    lines = [
        f"Temperature {cal['temperature']:.3f}. Bands: likely real below "
        f"{cal['likely_real_below']:.3f}, likely manipulated above {cal['likely_fake_above']:.3f}, "
        "uncertain in between.",
        "",
        "| Test set | Video ECE before | Video ECE after | Video log loss after |",
        "|---|---|---|---|",
    ]
    for name, m in results["sets"].items():
        if m.get("status") == "ok":
            lines.append(f"| {name} | {_f(m['video_ece_uncalibrated'])} | {_f(m['video_ece'])} "
                         f"| {_f(m['video_log_loss'])} |")  # fmt: skip
    return "\n".join(lines)


TESTED = {
    "ffpp_c23": "FaceForensics++ (c23): Deepfakes, Face2Face, FaceSwap, NeuralTextures",
    "ffpp_c40": "FaceForensics++ (c40, heavier compression): the same four methods",
    "celebdf_v2": "Celeb-DF v2: improved face swap",
    "dfdc": "DFDC (fixed 1,000-video subset): the DFDC competition's face swaps and reenactments",
    "deeperforensics": "DeeperForensics-1.0: DF-VAE face swap with real-world perturbations",
    "wilddeepfake": "WildDeepfake: fakes collected from the internet, methods unknown",
    "df40": "DF40: about 40 newer face swap, reenactment, generation and editing methods",
}


def _tested(results: dict[str, Any]) -> str:
    ok = [n for n, m in results["sets"].items() if m.get("status") == "ok"]
    lines = ["Tested on:", ""] + [f"- {TESTED.get(n, n)}" for n in ok]
    missing = [n for n, m in results["sets"].items() if m.get("status") != "ok"]
    if missing:
        lines += ["", f"Not tested (dataset not available at release): {', '.join(missing)}."]
    return "\n".join(lines)


def _hashes(hashes: dict[str, str]) -> str:
    rows = [f"| {name} | `{digest}` |" for name, digest in hashes.items()]
    return "\n".join(["| File | SHA-256 |", "|---|---|", *rows])


def fill_block(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- auto:{name}:start -->\n).*?(\n<!-- auto:{name}:end -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"model card has no auto:{name} block")
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def fill_card(
    template: str,
    results: dict[str, Any],
    calibration: dict[str, Any],
    hashes: dict[str, str],
    m5: dict[str, float] | None = None,
    fairness: dict[str, Any] | None = None,
) -> str:
    text = fill_block(template, "results", _results_table(results))
    if "<!-- auto:fairness:start -->" in text:
        from deepfake_detector.fairness import fairness_markdown

        body = (
            fairness_markdown(fairness)
            if fairness
            else "_To be filled: run python -m deepfake_detector.fairness on the evaluation._"
        )
        text = fill_block(text, "fairness", body)
    text = fill_block(text, "per_method", _per_method(results, m5))
    text = fill_block(text, "robustness", _robustness(results))
    text = fill_block(text, "calibration", _calibration(results, calibration))
    text = fill_block(text, "tested", _tested(results))
    return fill_block(text, "hashes", _hashes(hashes))


def validate(card: str, results: dict[str, Any], m5: dict[str, float] | None) -> list[str]:
    """Reasons the release is not ready; empty means ready to publish."""
    problems = []
    text = "\n" + card  # so a heading on the first line is found too
    for m in PLACEHOLDER.finditer(text):
        heading = text.rfind("\n## ", 0, m.start())
        section = text[heading + 4 : text.find("\n", heading + 4)] if heading >= 0 else "?"
        problems.append(f"unfinished section: {section}")
    if results["sets"].get(HEADLINE_SET, {}).get("status") != "ok":
        problems.append(f"headline result missing: {HEADLINE_SET} was not evaluated")
    if not m5:
        problems.append("unseen-method results (M5) missing: pass --m5")
    return sorted(set(problems))


# ----------------------------------------------------------------------------- stage, push


def stage(
    checkpoint: Path,
    results_path: Path,
    out: Path,
    template: Path = Path("docs/MODEL_CARD.md"),
    m5_path: Path | None = None,
) -> list[str]:
    """Build the release folder; return the list of problems blocking publication."""
    cal_path = checkpoint.parent / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path} missing: run calibrate first (S-3)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    export_weights(checkpoint, out)
    shutil.copy2(cal_path, out / "calibration.json")
    (out / "LICENSE").write_text(LICENSE_TEXT)
    files = ["model.safetensors", "config.json", "calibration.json", "LICENSE"]
    hashes = {f: sha256(out / f) for f in files}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in hashes.items()))

    results = json.loads(results_path.read_text())
    calibration = json.loads(cal_path.read_text())
    m5 = json.loads(m5_path.read_text()) if m5_path else None
    fair_path = results_path.parent / "fairness.json"
    fairness = json.loads(fair_path.read_text()) if fair_path.exists() else None
    card = fill_card(
        template.read_text(encoding="utf-8"), results, calibration, hashes, m5, fairness
    )
    (out / "README.md").write_text(card, encoding="utf-8")
    return validate(card, results, m5)


def push(
    out: Path, repo_id: str, problems: list[str], private: bool = False, api: Any = None
) -> str:
    """Upload the staged folder in one commit. Public releases need zero problems."""
    if problems and not private:
        raise RuntimeError("not ready to publish:\n- " + "\n- ".join(problems))
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release weights, model card and hashes together",
    )
    return str(getattr(info, "commit_url", info))


def main(argv: list[str] | None = None, confirm: Any = input) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True, help="results.json from evaluate")
    p.add_argument("--m5", type=Path, default=None, help='JSON {"Deepfakes": 0.91, ...}')
    p.add_argument("--card", type=Path, default=Path("docs/MODEL_CARD.md"))
    p.add_argument("--out", type=Path, default=Path("release"))
    p.add_argument("--push", action="store_true", help="upload to Hugging Face Hub")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--private", action="store_true", help="private repo; allows a draft card")
    args = p.parse_args(argv)

    problems = stage(args.checkpoint, args.results, args.out, args.card, args.m5)
    print(f"staged release in {args.out}")
    for line in (args.out / "SHA256SUMS").read_text().splitlines():
        print("  " + line)
    if problems:
        print("not ready to publish:")
        for pr in problems:
            print(f"  - {pr}")
    if not args.push:
        return 1 if problems else 0
    if not args.repo_id:
        p.error("--push needs --repo-id")
    if problems and not args.private:
        print("refusing to publish publicly until the problems above are fixed")
        return 1
    where = "privately" if args.private else "PUBLICLY"
    typed = confirm(
        f"This uploads {args.out} {where} to {args.repo_id}. Type the repo id to go on: "
    )
    if typed.strip() != args.repo_id:
        print("cancelled; nothing uploaded")
        return 1
    print(push(args.out, args.repo_id, problems, args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
