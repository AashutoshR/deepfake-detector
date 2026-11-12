# Evaluation protocols

| ID | Test | Train on | Test on | Target | Stretch |
|---|---|---|---|---|---|
| M1 | Same dataset | FF++ c23, all 4 methods | FF++ c23 test | >= 0.97 | >= 0.99 |
| M2 | Cross-dataset (headline) | FF++ c23 | Celeb-DF v2 test (518 videos) | >= 0.80 | >= 0.88 |
| M3 | Cross-dataset | FF++ c23 | DFDC subset | >= 0.70 | >= 0.75 |
| M4 | Cross-dataset | FF++ c23 | DeeperForensics, WildDeepfake | Report | >= 0.75 |
| M5 | Unseen method | 3 of 4 FF++ methods | The held-out 4th | Mean >= 0.85 | Mean >= 0.90 |
| M6 | Heavier compression | FF++ c23 | FF++ c40 | >= 0.85 | >= 0.90 |
| M7 | Calibration error (ECE) | FF++ c23 | Celeb-DF v2 | <= 0.10 | <= 0.05 |

Metric is video-level AUC unless stated (frame scores averaged per video). Headline numbers are
mean and std over 3 seeds with 95% bootstrap confidence intervals. Targets are revised after the
week 4 baselines.

Baselines (same pipeline): Xception, EfficientNet-B4, ViT-B/16 with plain fine-tuning.
