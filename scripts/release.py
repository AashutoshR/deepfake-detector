"""Stage the weights, model card and hashes, and optionally publish them together.

See deepfake_detector/release.py. Example:

    python scripts/release.py --checkpoint runs/<run>/best.pt \\
        --results outputs/eval/<run>/best/results.json --m5 outputs/m5.json --out release
"""

from deepfake_detector.release import main

if __name__ == "__main__":
    raise SystemExit(main())
