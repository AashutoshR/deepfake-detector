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

Once access is approved, `scripts/get_data.py` runs the official FF++ script, unpacks the
Celeb-DF and DFDC archives you received, arranges the other test sets, and counts every file:

```bash
python scripts/get_data.py disk
python scripts/get_data.py check
```

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

## Training

```bash
python -m deepfake_detector.train                                   # ViT-B/16 + LoRA, defaults
python -m deepfake_detector.train train.leave_out_method=Deepfakes train.run_name=lomo_deepfakes
python -m deepfake_detector.train model.arch=xception train.run_name=xception
python -m deepfake_detector.train --resume runs/<run>/last.pt
tensorboard --logdir runs
```

Each run folder in `runs/` holds the merged config, library versions and git commit, TensorBoard
logs, `metrics.jsonl`, `last.pt` (saved every 30 minutes and every epoch) and `best.pt` (best
FF++ validation video AUC). Test sets are never used during training.

## Calibration and testing

```bash
python -m deepfake_detector.calibrate --checkpoint runs/<run>/best.pt
python -m deepfake_detector.evaluate --checkpoint runs/<run>/best.pt
python -m deepfake_detector.evaluate --checkpoint runs/<run>/best.pt --robustness
```

Calibration fits one temperature and the three band cut-offs on FF++ validation only, and saves
`calibration.json` next to the checkpoint. Testing picks it up automatically and writes to
`outputs/eval/<run>/<checkpoint>/`: a results table (`results.md`, `results.json`) with video
AUC and its 95% bootstrap confidence interval, frame AUC, EER, accuracy, log loss and ECE before
and after calibration for every test set, plus ROC curves, calibration charts, per-method AUC
on FF++, every frame score, and with `--robustness` the AUC under JPEG, blur, downscaling,
noise and contrast loss at rising strength. Test sets not downloaded yet show as "not
available".

## Predict and local demo

```bash
predict --input clip.mp4 --checkpoint runs/<run>/best.pt
predict --input face.jpg --checkpoint runs/<run>/best.pt --heatmap heat.png
```

Prints JSON with the calibrated probability, one of three bands (likely real, uncertain,
likely manipulated), per-frame scores and a note that the score is an estimate, not proof.
The uncertain band is always at least 20 points wide, even when validation data separates
cleanly, so a score near the middle never gets a confident label.

The demo page runs on this computer only:

```bash
set DFD_CHECKPOINT=runs\<run>\best.pt
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, then drop in a video or image. It shows the score, band, a score
for every frame, and an attention heatmap on the most suspicious face. Each upload is deleted
straight after scoring, and nothing about it is stored or logged. Without a GPU it runs on the
CPU, more slowly.

## GPU budget (RTX 4060 Laptop, 8 GB)

Measured with `python scripts/bench_gpu.py` (batch 16, bf16, 224 x 224):

| Setting | Trainable params | Peak memory | Images/s |
|---|---|---|---|
| ViT-B/16 LoRA (default) | 0.6M | 1.3 GB | 185 |
| ViT-B/16 LoRA, gradient checkpointing | 0.6M | 0.6 GB | 136 |
| ViT-B/16 full fine-tuning, checkpointing | 85.8M | 1.6 GB | 97 |
| ViT-B/16 last layer only | 7.1M | 0.5 GB | 335 |
| Xception full | 20.8M | 1.4 GB | 168 |
| EfficientNet-B4 full | 17.6M | 0.6 GB | 74 |

ViT-B/16 inference: about 2.6 ms per face. A laptop GPU can slow down when hot, so long runs
should be plugged in and well ventilated.

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
