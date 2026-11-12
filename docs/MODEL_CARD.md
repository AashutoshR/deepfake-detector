---
license: cc-by-nc-4.0
library_name: timm
pipeline_tag: image-classification
tags:
  - deepfake-detection
  - face-forensics
  - vision-transformer
  - lora
  - calibration
metrics:
  - roc_auc
---

# deepfake-detector

A Vision Transformer (ViT-B/16) fine-tuned with LoRA on FaceForensics++ to estimate how likely a
face video or image is to have been manipulated, built to keep working on datasets and
fake-making methods it never saw in training.

> **Scores are estimates from a statistical model, not proof that a video is real or fake.**
> Every output is a calibrated probability and one of three bands: likely real, uncertain,
> likely manipulated. Never use it as the only basis for a decision about a real person.

## Model details

| | |
|---|---|
| Architecture | ViT-B/16 (about 86M parameters), LoRA on attention, linear head on the CLS token |
| Input | 224 x 224 face crop, 1.3x the detected face box (SFD detector, 68 landmarks) |
| Output | Probability of manipulation per frame; video score = mean of frame scores |
| Calibration | One temperature fitted on FF++ validation; band cut-offs fitted on FF++ validation |
| Training | FF++ c23 official train split, self-blended images, degradation augmentation, bf16 |
| Code | MIT license, <https://github.com/AashutoshR/deepfake-detector> |
| Weights | CC BY-NC 4.0, research and non-commercial use only |

## Intended use

- Research on deepfake detection, especially how detectors generalize to unseen datasets and
  methods.
- A second opinion inside a human review process, alongside other evidence such as provenance
  and source checks.
- Teaching and demonstrating what current detectors can and cannot do.

## Out-of-scope uses

- Deciding on its own whether a real person's video is authentic.
- Legal, journalistic, employment, moderation or law-enforcement decisions without human
  review and other evidence.
- Surveillance or identifying people. The model does not recognize who is in a video.
- Audio deepfakes, fully generated images that are not faces, and videos with no visible face.
- Mobile or real-time use; it was not built or tested for either.
- Any claim that a video is proven real or proven fake.

## How to use

```bash
git clone https://github.com/AashutoshR/deepfake-detector
cd deepfake-detector
pip install -r requirements.txt && pip install -e .
predict --input clip.mp4 --checkpoint <downloaded folder>/model.safetensors
```

The output is JSON with the probability, band, per-frame scores and the not-proof note. The
local demo page (`uvicorn api.server:app`) shows the same result with a heatmap.

## Training data

FaceForensics++ (Rössler et al., 2019), c23 compression, official 720 / 140 / 140 video split
by identity: 1,000 real YouTube videos and four manipulation methods (Deepfakes, Face2Face,
FaceSwap, NeuralTextures). Half of the real training crops are turned into self-blended
images (a face blended with an altered copy of itself) so the model learns blending marks in
general instead of one generator's style. The model and its band cut-offs were chosen on FF++
validation only; no test set was used for any choice.

## Results (S-1, S-2)

Video-level AUC with 95% bootstrap confidence intervals. Cross-dataset results are in the same
table, with the same prominence, as the same-dataset result: they are the numbers that matter
for real use.

<!-- auto:results:start -->
_Filled by scripts/release.py from the evaluation results._
<!-- auto:results:end -->

<!-- auto:per_method:start -->
_Filled by scripts/release.py from the evaluation results._
<!-- auto:per_method:end -->

## Robustness

<!-- auto:robustness:start -->
_Filled by scripts/release.py from the evaluation results._
<!-- auto:robustness:end -->

Degradations were applied to the face crops, which approximates but is not identical to
degrading the whole video.

## Calibration (S-3)

A confident wrong answer about a real person's video can do real harm, so scores are calibrated
before they are shown. One temperature is fitted on FF++ validation frames; it changes how
confident scores are, never their order (AUC). The uncertain band is always at least 20
points wide, so scores near the middle never get a confident label.

<!-- auto:calibration:start -->
_Filled by scripts/release.py from the evaluation results._
<!-- auto:calibration:end -->

Calibration fitted on FF++ does not carry over perfectly to other datasets: check the ECE after
calibration for each test set above before trusting the bands on a new kind of video.

## Fairness (S-5)

Results per demographic group use published labels only; this project never infers anyone's
demographics. A false alarm (a real video put in the likely-manipulated band) is the most
harmful error for the person shown, so it is reported per group alongside AUC.

<!-- auto:fairness:start -->
_Filled by scripts/release.py from fairness.json._
<!-- auto:fairness:end -->

_To be filled: what the groups show, why any flagged gap (above 5 AUC points) may arise, and
what a user should keep in mind._

## Failure cases

_To be filled: what the heatmap report shows about false alarms and misses on every test set,
with patterns (for example lighting, pose, compression, skin tone, makeup, glasses)._

## Limitations and tested methods (S-10)

<!-- auto:tested:start -->
_Filled by scripts/release.py from the evaluation results._
<!-- auto:tested:end -->

Not tested, and likely to be missed: face generators and editors newer than the test sets above
(for example recent diffusion-based face swaps and talking-head generators), heavily filtered
or re-encoded social media video beyond the robustness sweep, partial edits to small regions,
and any manipulation of audio. A low score does not mean a video is authentic.

## Misuse (S-6)

A public detector can be used as a target: someone can tweak a fake until it scores low, then
point to the low score as if it were proof. We release anyway, for three reasons:

1. Strong detectors already circulate privately; a public, documented one lets researchers
   measure and improve generalization openly, including this model's weak spots.
2. The model is released with its failure cases, calibration and limits, so a low score is
   presented as "not detected by this model", never as "real".
3. A detector like this one takes one consumer GPU and research datasets any lab can request,
   so withholding the weights would slow honest research more than it would slow anyone
   tuning fakes.

Please do not present this model's output as proof in any public claim about a real person.

## Dataset ethics (S-9)

Every dataset is used under its own license and only for research. No face crops, frames or
videos from any dataset are included in this repository, the weights repository, or this card.
Many people in these datasets did not consent to appearing in them; the datasets were built
from public videos for research. That is one more reason this model must not be used to make
claims about specific people.

## Licenses (S-7)

- Code: MIT.
- Weights: CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>), because
  FaceForensics++ and Celeb-DF allow research and non-commercial use only. Using the weights
  also means respecting the terms of the datasets they were trained on.

## Integrity (S-8)

Check a download with `sha256sum` (Linux) or `Get-FileHash -Algorithm SHA256` (Windows).

<!-- auto:hashes:start -->
_Filled by scripts/release.py._
<!-- auto:hashes:end -->

## Citation

If you use this model, please cite the repository and the datasets it was trained and tested
on (FaceForensics++, Celeb-DF, DFDC, DeeperForensics-1.0, WildDeepfake), and the
self-blended images method (Shiohara and Yamasaki, CVPR 2022).
