# Ablations

Each ablation changes one thing from the final model and is scored on M1, M2 and M5 with 1 seed.
At least 6 of the 8 are run. The write-up explains why the winning settings help, with evidence.

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
