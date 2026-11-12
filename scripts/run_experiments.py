"""Train and test the final model and the three baselines over 3 seeds; build the headline table.

See deepfake_detector/experiments.py. Examples:

    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --only final --seeds 0 1 2
"""

from deepfake_detector.experiments import main

if __name__ == "__main__":
    main()
