# deepfake-detector

A Vision Transformer (ViT-B/16) trained on FaceForensics++ that aims to keep catching deepfakes
from datasets and fake-making methods it never saw in training.

> **Status:** early setup. No results yet.
>
> Scores from this tool are estimates, not proof that a video is real or fake.

## The problem

Most detectors score about 0.99 AUC on the dataset they trained on, then drop to roughly 0.65 to
0.75 on a different one. In the real world the fake-making method is almost never one you trained
on. This project attacks that gap.

**Headline metric:** video-level AUC on Celeb-DF v2 after training only on FaceForensics++
(target 0.80 or higher).

## Scope

- In: face videos and images, real vs manipulated; training on FaceForensics++; testing on
  Celeb-DF v2, DFDC, DeeperForensics-1.0, WildDeepfake and optionally DF40; a CLI and a local
  web demo; public weights with a model card.
- Out: audio deepfakes, fully generated non-face images, mobile or real-time use, and any claim
  that the tool proves a video is real or fake.

## Setup (Windows, NVIDIA GPU)

1. Install Python 3.11 and the latest NVIDIA driver.
2. Create a virtual environment:
   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```
3. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
4. Check the GPU (should print `True` and the GPU name):
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
5. Install the rest and the package:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Datasets

No dataset can be downloaded freely; each requires accepting its license first. Datasets live in
`data/`, which git ignores. No dataset file is ever committed or re-shared. See
[docs/datasets.md](docs/datasets.md) for access steps and license terms.

## Preprocessing

Extracts 32 evenly spaced frames per video, detects the main face with the SFD detector, saves
224 x 224 crops at 1.3x the face box, and writes a Parquet manifest. Each video gets its own
shard, so a stopped run resumes where it left off. Videos with no face are listed in
`skipped.parquet` with a reason, never dropped silently.

```bash
python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp --out data
python -m deepfake_detector.data.preprocess --dataset ffpp --raw data/raw/ffpp --out data --compression c40 --splits test
python -m deepfake_detector.data.preprocess --dataset celebdf_v2 --raw data/raw/celebdf_v2 --out data
```

Add `--limit 20` for a quick trial run. The manifest is checked automatically so no video or
person appears in two splits.

## Layout

```
configs/               YAML configs: data, model, train, eval, ablations
src/deepfake_detector/
  data/                extract, faces, datasets, sbi, augment
  models/              vit, baselines
  train.py  evaluate.py  calibrate.py  explain.py  predict.py
api/server.py          FastAPI app and local demo page
scripts/               dataset download, ablation launcher, report builder
tests/
docs/                  datasets, protocols, ablations, MODEL_CARD
data/                  git-ignored: raw videos, face crops, manifests
```

## License

Code: MIT. Model weights: non-commercial research license (training data allows research use
only). See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
