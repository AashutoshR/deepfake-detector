# Blog post draft: "My deepfake detector scored 99%. Then I tested it properly."

Plain-language companion to the paper, published together with the weights. Numbers in
[brackets] are placeholders until the experiments finish. Target length: 1,200 to 1,500 words,
two figures.

## 1. The 99% trap (hook, about 150 words)

Most deepfake detectors look nearly perfect in their own papers: around 99% on the benchmark
they trained on. Point the same detector at fakes from somewhere else and it often drops to
65 to 75%, not far above guessing. Real fakes never come from your training set, so the first
number is the one that misleads.

## 2. Why detectors overfit to their training fakes (about 200 words)

- Each fake-making tool leaves its own fingerprint: a color mismatch, a blur pattern, a
  compression trace.
- A detector trained on four tools can learn those four fingerprints instead of "what a
  manipulated face looks like".
- Analogy: a spam filter that only learned the four spammers it saw.

## 3. Two ideas that help (about 300 words)

- **Teach the seam, not the tool.** Self-blended images: take a real face, slightly alter a copy,
  paste it back over itself. The only thing to learn is the blending seam that almost every face
  swap leaves. Figure: real face, altered copy, blend mask, result (use a face we are allowed to
  show, never one from the datasets).
- **Start from a model that already sees well, and change it as little as possible.** A Vision
  Transformer pretrained on millions of images, fine-tuned with LoRA: [0.6M] of [86M] weights.
  Fewer weights to change means fewer ways to memorize the training fakes.

## 4. Testing it properly (about 300 words)

- Trained only on FaceForensics++; tested on Celeb-DF, DFDC, DeeperForensics and WildDeepfake.
- Held out one fake-making method at a time to test methods it never saw.
- Three seeds, confidence intervals, and three baselines trained the same way.
- Figure: one chart, same-dataset vs cross-dataset AUC for each model.
- Result: [X.XX] on Celeb-DF vs [X.XX] for the best baseline; [what did not work].

## 5. A score is not a verdict (about 200 words)

- Why a confident wrong answer about a real person is worse than "uncertain".
- Calibration: when the model says 80%, it should be right about 80% of the time.
- Three bands instead of real/fake, and why the uncertain band is never allowed to vanish.
- Fairness: false alarms per group, and what we found [result].

## 6. What it cannot do (about 150 words)

- Newer generators it was never tested on; heavy re-encoding; audio.
- A low score means "not detected by this model", never "real".
- Why we released it anyway (short version of the model card's misuse section).

## 7. Try it (about 50 words)

Links: GitHub repo, Hugging Face weights, paper. Runs on a laptop GPU or CPU.
