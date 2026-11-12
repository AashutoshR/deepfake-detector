# Reproducing the results

From a fresh clone to the headline numbers, in order. Every step is resumable: if a run stops
(crash, sleep, power), run the same command again and it continues where it left off.

GPU hours are estimates for one RTX 4060 Laptop GPU (8 GB), from the measured training speed in
the README. A desktop GPU will be faster.

## 1. Environment (about 15 minutes)

Python 3.10 to 3.12 on Windows or Linux, with an NVIDIA driver.

```bash
git clone https://github.com/AashutoshR/deepfake-detector
cd deepfake-detector
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

All tests should pass. `python scripts/bench_gpu.py` shows memory and speed on your GPU.

## 2. Datasets (days to weeks, waiting for approval)

Request access under your own name to each dataset (see [datasets.md](datasets.md)). When the
approvals arrive:

```bash
python scripts/get_data.py disk
python scripts/get_data.py ffpp --script faceforensics_download_v4.py
python scripts/get_data.py celebdf --zip Celeb-DF-v2.zip
python scripts/get_data.py dfdc --zip dfdc_train_part_0.zip --n 1000
python scripts/get_data.py check
```

`check` must show OK for every dataset you downloaded before going on.

## 3. Face crops (several hours, CPU and GPU)

```bash
python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp --limit 20
```

Open a few crops in `data/crops/ffpp/` and check they show centred faces. Then run everything:

```bash
python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp
python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp --compression c40 --splits test
python -m deepfake_detector.data.preprocess --dataset celebdf_v2 --raw data/raw/celebdf_v2
python -m deepfake_detector.data.preprocess --dataset dfdc --raw data/raw/dfdc
```

Each run ends with the number of crops and skipped videos, with reasons. A skip rate above a
few percent means something is wrong with the videos or the detector.

## 4. Headline experiments (about 197 GPU hours in full)

The final model and the three baselines, 3 seeds each, each with one leave-one-method-out run
per FF++ method for M5:

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --only final
python scripts/run_experiments.py --only plain_vit xception efficientnet_b4
```

| Part | Runs | GPU hours |
|---|---|---|
| Final model, 3 seeds, 20 epochs | 15 | about 52 |
| Three baselines, 3 seeds, 10 epochs | 45 | about 145 |
| Everything with M5 on NeuralTextures only | 24 | about 79 |

Results land in `outputs/experiments/results.md`: mean ± std over seeds for M1 to M7, whether
each target was met, and whether the final model beats every baseline on M2 and M5.

## 5. Robustness, heatmaps and fairness (a few GPU hours)

On the final model, seed 0:

```bash
python -m deepfake_detector.evaluate --checkpoint runs/experiments/exp_final_s0/best.pt --robustness --out outputs/eval/final_s0
python -m deepfake_detector.explain --checkpoint runs/experiments/exp_final_s0/best.pt --eval-dir outputs/eval/final_s0
python -m deepfake_detector.fairness --eval-dir outputs/eval/final_s0 --labels celebdf_v2=data/labels/celebdf_v2.csv
```

Fairness needs published demographic labels converted to a CSV keyed by this project's video
ids (see the docstring of `deepfake_detector/fairness.py`).

## 6. Ablations (about 31 to 77 GPU hours)

```bash
python scripts/run_ablations.py --dry-run
python scripts/run_ablations.py --epochs 8 --m5-methods NeuralTextures
```

See [ablations.md](ablations.md).

## 7. Release

Finish the hand-written sections of [MODEL_CARD.md](MODEL_CARD.md) (fairness discussion,
failure cases), then:

```bash
python scripts/release.py --checkpoint runs/experiments/exp_final_s0/best.pt --results outputs/eval/final_s0/results.json --m5 m5.json --out release
```

The release stays blocked until nothing is missing. See the README's Release section.

## What "reproduced" means

A fresh clone following these steps should land within the reported standard deviation of the
headline numbers (NFR-7). Exact numbers can differ slightly between GPUs and library versions;
every run saves its config, library versions and git commit in `runs/<run>/env.json` and
`config.yaml`, so differences can be traced.
