# Model card: deepfake-detector

> Draft. This card ships together with the weights on Hugging Face Hub, never later.

## Model details
ViT-B/16 fine-tuned with LoRA to score face crops as real or manipulated. Weights license:
non-commercial research use. Code license: MIT.

## Intended use
Research on deepfake detection and generalization. Outputs a calibrated probability and one of
three bands (likely real, uncertain, likely manipulated), with a note that the score is an
estimate, not proof (S-4).

## Out-of-scope uses
Deciding on its own whether a real person's video is authentic; legal, journalistic or
moderation decisions without human review; audio deepfakes; non-face generated images.

## Training data
FaceForensics++ c23 (official splits), used under its research license.

## Results (S-1, S-2)
Cross-dataset results are reported with the same prominence as same-dataset results.
_To be filled._

## Calibration (S-3)
Temperature scaling fitted on FF++ validation; ECE before and after on every test set.
_To be filled._

## Fairness (S-5)
Results per demographic group where published labels exist; gaps above 5 AUC points are flagged.
_To be filled._

## Failure cases
_To be filled, with examples from every test set._

## Misuse (S-6)
A public detector can be used to tweak fakes until they pass it. _Reasoning for release to be
written._

## Tested and untested methods (S-10)
_List of fake-making methods tested._ Newer, untested generators may not be caught.

## Dataset ethics (S-9)
Every dataset license is respected and no faces are re-shared. Not everyone in these datasets
consented to appear in them.

## Integrity (S-8)
| File | SHA-256 |
|---|---|
| _to be filled_ | |
