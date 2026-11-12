# Ablations

Each ablation changes one thing from the final model and is scored on M1, M2 and M5 with 1 seed.
At least 6 of the 8 are run. The write-up explains why the winning settings help, with evidence.

## Running them

```bash
python scripts/run_ablations.py --dry-run                        # runs needed and GPU hours
python scripts/run_ablations.py --only A1 A3 A4 --epochs 8
python scripts/run_ablations.py --m5-methods NeuralTextures      # cheaper M5
python scripts/run_ablations.py --gather-only                    # rebuild the table
```

Options equal to the base config share one run, and A8 re-scores saved frame scores without
training. Finished runs are skipped and stopped runs resume, so the launcher can be stopped and
started again at any time. Results go to `outputs/ablations/ablations.md` and `.json`.

Cost on the RTX 4060 Laptop at 8 epochs (estimate from the measured speed):

| M5 held-out methods | Runs | GPU hours |
|---|---|---|
| All 4 (as planned) | 55 | about 77 |
| NeuralTextures only | 22 | about 31 |

A7 (c23 mixed with c40) needs c40 crops for the whole FF++ training split, about 16 GB more
disk.

| # | What changes | Options | M1 | M2 | M5 | Notes |
|---|---|---|---|---|---|---|
| A1 | Starting weights | ImageNet-21k, CLIP, DINOv2 | | | | |
| A2 | Fine-tuning | LoRA, full, last layer only | | | | |
| A3 | Synthetic fakes | None, SBI only, SBI plus FF++ fakes | | | | |
| A4 | Augmentation | Basic, full degradation set | | | | |
| A5 | Token pooling | CLS, mean of patch tokens | | | | |
| A6 | Extra head | Real/fake only, also predict method | | | | |
| A7 | Training compression | c23 only, c23 mixed with c40 | | | | |
| A8 | Video score | Mean, median, top-k | | | | |
