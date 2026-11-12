"""Measure peak GPU memory and speed for each model setting on random data (NFR-1, NFR-4).

No dataset or weight download needed: memory and speed do not depend on the weight values.

    python scripts/bench_gpu.py
    python scripts/bench_gpu.py --batch 16 --steps 30
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from deepfake_detector.models import build_model
from deepfake_detector.models.common import count_parameters

SETTINGS = [
    ("ViT-B/16 LoRA, grad ckpt", {"arch": "vit", "finetune": {"mode": "lora"}}, True),
    ("ViT-B/16 LoRA, no ckpt", {"arch": "vit", "finetune": {"mode": "lora"}}, False),
    ("ViT-B/16 full, grad ckpt", {"arch": "vit", "finetune": {"mode": "full"}}, True),
    ("ViT-B/16 last layer", {"arch": "vit", "finetune": {"mode": "last_layer"}}, True),
    ("Xception full", {"arch": "xception"}, True),
    ("EfficientNet-B4 full", {"arch": "efficientnet_b4"}, True),
]


def bench_train(cfg: dict, ckpt: bool, batch: int, steps: int) -> tuple[float, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model({**cfg, "pretrained": "none", "grad_checkpointing": ckpt}, False).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.rand(batch, 3, 224, 224, device="cuda")
    y = torch.randint(0, 2, (batch,), device="cuda").float()
    for i in range(steps + 3):
        if i == 3:  # skip warm-up steps
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss_fn(out["logit"].float(), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    sec_per_step = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    trainable, _ = count_parameters(model)
    del model, opt
    return peak, sec_per_step, trainable / 1e6


@torch.no_grad()
def bench_infer(batch: int = 32, steps: int = 20) -> float:
    model = build_model({"arch": "vit", "pretrained": "none"}, False).cuda().eval()
    x = torch.rand(batch, 3, 224, 224, device="cuda")
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps / batch * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    assert torch.cuda.is_available(), "no CUDA GPU found"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total:.1f} GB), batch {args.batch}, bf16\n")
    print(f"{'setting':28} {'trainable M':>11} {'peak GB':>8} {'s/step':>7} {'img/s':>6}")
    for name, cfg, ckpt in SETTINGS:
        try:
            peak, sps, tm = bench_train(cfg, ckpt, args.batch, args.steps)
            print(f"{name:28} {tm:11.1f} {peak:8.2f} {sps:7.3f} {args.batch / sps:6.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:28} {'out of memory':>35}")
            torch.cuda.empty_cache()
    print(f"\nViT-B/16 inference: {bench_infer():.1f} ms per face (batch 32, bf16)")


if __name__ == "__main__":
    main()
