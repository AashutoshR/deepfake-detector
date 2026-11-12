"""Heatmaps: attention rollout (FR-X1), Grad-CAM (FR-X2) and a report on right and wrong
predictions for each test set (FR-X3).

Rollout (Abnar and Zuidema, 2020) multiplies the head-averaged attention of every layer, with
the residual connection added, to estimate how much each image patch feeds the final score.
Grad-CAM (Selvaraju et al., 2017) weights the last feature map by the gradient of the fake
score, and also works for the CNN baselines. Both show where the model looks, not why: a
heatmap is a hint for a human, not evidence.

    python -m deepfake_detector.explain --checkpoint runs/<run>/best.pt \\
        --eval-dir outputs/eval/<run>/best
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from deepfake_detector.models.common import Detector

log = logging.getLogger(__name__)


def attention_rollout(model: Detector, x: torch.Tensor) -> np.ndarray:
    """Return heatmaps (B, gh, gw) in [0, 1] for 0 to 1 RGB input x (B, 3, H, W)."""
    backbone: Any = model.backbone
    # CNNs such as EfficientNet also have .blocks, so check for the patch embedding instead.
    if not hasattr(backbone, "patch_embed"):
        raise ValueError("attention rollout needs a ViT backbone")
    blocks = backbone.blocks

    attns: list[torch.Tensor] = []

    def make_hook(attn: Any) -> Any:
        def hook(_: Any, __: Any, out: torch.Tensor) -> None:
            b, n, _c = out.shape
            qkv = out.reshape(b, n, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
            a = (q.float() @ k.float().transpose(-2, -1)) * attn.scale
            attns.append(a.softmax(-1).mean(1))

        return hook

    handles = [blk.attn.qkv.register_forward_hook(make_hook(blk.attn)) for blk in blocks]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    n = attns[0].shape[-1]
    eye = torch.eye(n, device=attns[0].device)
    rollout = eye.expand_as(attns[0]).clone()
    for a in attns:
        a = a + eye
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout

    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    if getattr(backbone, "global_pool", "token") == "avg":
        scores = rollout[:, prefix:, prefix:].mean(1)  # mean-pooled model: average patch rows
    else:
        scores = rollout[:, 0, prefix:]  # CLS token row
    g = int(math.isqrt(scores.shape[-1]))
    heat = scores[:, : g * g].reshape(-1, g, g)
    lo = heat.amin(dim=(1, 2), keepdim=True)
    hi = heat.amax(dim=(1, 2), keepdim=True)
    return ((heat - lo) / (hi - lo + 1e-8)).cpu().numpy()


def overlay(crop: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (any size, 0 to 1) over an RGB uint8 crop."""
    h, w = crop.shape[:2]
    up = cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(up, 0, 1) * 255), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(crop * (1 - alpha) + colored * alpha)


# ----------------------------------------------------------------------------- Grad-CAM


class _FakeLogit(nn.Module):
    """Wraps a Detector so Grad-CAM sees a (B, 1) output: the fake logit."""

    def __init__(self, model: Detector) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["logit"].unsqueeze(-1)


def _target_layer(backbone: Any) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor] | None]:
    """Last feature layer for Grad-CAM, plus a reshape for ViT token outputs."""
    if hasattr(backbone, "patch_embed"):
        prefix = int(getattr(backbone, "num_prefix_tokens", 1))

        def reshape(t: torch.Tensor) -> torch.Tensor:
            tokens = t[:, prefix:, :]
            g = int(math.isqrt(tokens.shape[1]))
            return tokens[:, : g * g].reshape(t.shape[0], g, g, -1).permute(0, 3, 1, 2)

        return backbone.blocks[-1].norm1, reshape
    if hasattr(backbone, "conv_head"):  # EfficientNet
        return backbone.bn2, None
    if hasattr(backbone, "act4"):  # Xception
        return backbone.act4, None
    raise ValueError(f"no Grad-CAM target layer known for {type(backbone).__name__}")


def grad_cam(model: Detector, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM heatmaps (B, H, W) in [0, 1] for the fake score of 0 to 1 RGB input x."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer, reshape = _target_layer(model.backbone)
    was_training = model.training
    model.eval()
    try:
        with GradCAM(_FakeLogit(model), [layer], reshape_transform=reshape) as cam:
            targets = [ClassifierOutputTarget(0) for _ in range(len(x))]
            heat = cam(x.clone().requires_grad_(True), targets)
    finally:
        model.train(was_training)
    return np.asarray(heat, dtype=np.float32)


# ----------------------------------------------------------------------------- report

CASES = ("fake, caught", "real, cleared", "real, flagged (false alarm)", "fake, missed")


def pick_examples(frames: pd.DataFrame, n: int = 4, threshold: float = 0.5) -> pd.DataFrame:
    """The n most confident frames of each case, at most one frame per video.

    Wrong predictions are always included when they exist, so failures are shown (S-2).
    """
    fake, real = frames["label"] == 1, frames["label"] == 0
    high = frames["prob"] >= threshold
    selections = {
        CASES[0]: frames[fake & high].sort_values("prob", ascending=False),
        CASES[1]: frames[real & ~high].sort_values("prob"),
        CASES[2]: frames[real & high].sort_values("prob", ascending=False),
        CASES[3]: frames[fake & ~high].sort_values("prob"),
    }
    rows = [s.drop_duplicates("video_id").head(n).assign(case=c) for c, s in selections.items()]
    return pd.concat(rows, ignore_index=True)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"missing crop: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def heatmap_report(
    checkpoint: str | Path, eval_dir: str | Path, n: int = 4, data_root: str | Path | None = None
) -> list[Path]:
    """Write heatmaps_<set>.png and heatmaps.md into eval_dir from its saved frame scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deepfake_detector.inference import load_detector

    eval_dir = Path(eval_dir)
    results = json.loads((eval_dir / "results.json").read_text())
    root = Path(data_root or results["data_root"])
    model, cfg = load_detector(checkpoint)
    device = next(model.parameters()).device
    is_vit = cfg["model"].get("arch", "vit") == "vit"
    written: list[Path] = []
    md = [
        "# Heatmaps on right and wrong predictions",
        "",
        "Most confident frames per case, one per video. Heatmaps show where the model looked, "
        "not why; they are hints, not evidence.",
        "",
    ]
    for name, res in results["sets"].items():
        path = eval_dir / f"{name}_frames.parquet"
        if res.get("status") != "ok" or not path.exists():
            continue
        ex = pick_examples(pd.read_parquet(path), n)
        if ex.empty:
            continue
        crops = [_read_rgb(root / p) for p in ex["crop_path"]]
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(device)
        rows = [
            ("crop", crops),
            ("Grad-CAM", [overlay(c, h) for c, h in zip(crops, grad_cam(model, x), strict=True)]),
        ]
        if is_vit:
            rolls = attention_rollout(model, x)
            rows.append(("rollout", [overlay(c, h) for c, h in zip(crops, rolls, strict=True)]))

        fig, axes = plt.subplots(
            len(rows), len(ex), figsize=(1.9 * len(ex), 2.1 * len(rows)), squeeze=False
        )
        for i, (label, images) in enumerate(rows):
            axes[i][0].set_ylabel(label, fontsize=7)
            for j, img in enumerate(images):
                axes[i][j].imshow(img)
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
        for j, r in enumerate(ex.itertuples(index=False)):
            axes[0][j].set_title(f"{r.case}\np={r.prob:.2f} {r.method}", fontsize=6)
        fig.tight_layout()
        out = eval_dir / f"heatmaps_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
        counts = ex["case"].value_counts()
        md += [f"## {name}", "", f"![{name}]({out.name})", ""]
        md += [", ".join(f"{c}: {int(counts.get(c, 0))}" for c in CASES), ""]
    (eval_dir / "heatmaps.md").write_text("\n".join(md) + "\n")
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--eval-dir", required=True, type=Path, help="output folder of evaluate")
    p.add_argument("--n", type=int, default=4, help="examples per case")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for out in heatmap_report(args.checkpoint, args.eval_dir, args.n, args.data_root):
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
