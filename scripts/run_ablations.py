"""Run the ablation grid A1 to A8 and gather one results table (FR-E9).

See deepfake_detector/ablations.py for details. Examples:

    python scripts/run_ablations.py --dry-run
    python scripts/run_ablations.py --only A1 A3 --epochs 8
"""

from deepfake_detector.ablations import main

if __name__ == "__main__":
    main()
