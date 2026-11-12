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

## Getting each dataset

Run these after your own request is approved. `scripts/get_data.py` never fetches a dataset by
itself: it runs the script or unpacks the archive you received, puts it in `data/raw/`, and
counts the files. Every step ends with the same check table.

```bash
python scripts/get_data.py disk                                    # free space vs the plan
python scripts/get_data.py ffpp --script faceforensics_download_v4.py
python scripts/get_data.py celebdf --zip Celeb-DF-v2.zip
python scripts/get_data.py dfdc --zip dfdc_train_part_0.zip --n 1000
python scripts/get_data.py folder --dataset wilddeepfake --real <real dir> --fake <fake dir> --method wild
python scripts/get_data.py check
```

- **FaceForensics++:** runs the official script from the approval email once per item
  (original videos and the four methods, c23 and c40, videos only), so it skips the large
  DeepFakeDetection and FaceShifter sets. The script asks you to accept its terms, so run it in
  your own terminal. It then fetches the official split lists (public, on the FF++ GitHub).
  The check expects 1,000 videos per item and a 720 / 140 / 140 split.
- **Celeb-DF v2:** unpacks the zip and checks for 518 test videos.
- **DFDC:** takes a fixed subset (seed 0) from one training part: up to 500 real videos, the
  rest fake, 1,000 in total. Only the chosen videos are unpacked, and the archive can be
  deleted afterwards.
- **DeeperForensics-1.0, WildDeepfake, DF40:** their layouts differ by release, so `folder`
  links your unpacked real and fake folders into `real/` and `fake/<method>/`. Videos, single
  images and folders of frames all work. Nothing is moved or deleted.

Archives are unpacked safely: any file path that tries to leave the target folder stops the
unpacking. Sizes and access steps in the plan are approximate, so confirm them on each page.

## Disk budget

147 GB free on C (checked 2026-11-12), against about 100 GB needed. If space runs short:

- FF++: c23 for all splits; c40 for the 140 test videos only (M6).
- Delete raw videos once their face crops pass the count check.
- Add DeeperForensics, WildDeepfake and DF40 one at a time.

## Splits

- FF++ official splits: 720 / 140 / 140 videos.
- No person appears in two splits; an automated test checks this.
- Test sets are never used to pick a model, a threshold or when to stop training.
