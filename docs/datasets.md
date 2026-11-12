# Datasets

No dataset is downloaded until its license is accepted under the requester's own name. Nothing
from these datasets is ever committed to this repository or re-shared. Not everyone shown in
these datasets consented to it (S-9).

| Priority | Dataset | Role | Status | Date requested | Date approved | License terms |
|---|---|---|---|---|---|---|
| 1 | FaceForensics++ | Train, val, same-dataset test | Not requested | | | |
| 1 | Celeb-DF v2 | Main cross-dataset test (M2) | Not requested | | | |
| 2 | DFDC (about 1,000 video subset) | Cross-dataset test (M3) | Not requested | | | |
| 3 | DeeperForensics-1.0 | Cross-dataset, robustness (M4) | Not requested | | | |
| 3 | WildDeepfake | In-the-wild test (M4) | Not requested | | | |
| 4 | DF40 (optional) | Unseen modern methods | Not requested | | | |

## Disk budget (about 50 GB free)

- FF++: c23 for all splits; c40 for the 140 test videos only (M6).
- Delete raw videos once their face crops pass the count check.
- DeeperForensics, WildDeepfake and DF40 are added one at a time as space allows.

## Splits

- FF++ official splits: 720 / 140 / 140 videos.
- No person appears in two splits; an automated test checks this.
- Test sets are never used to pick a model, a threshold or when to stop training.
