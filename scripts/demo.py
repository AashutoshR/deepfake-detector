"""Start the local demo and open it in the browser.

    python scripts/demo.py                                  # newest best.pt under runs/
    python scripts/demo.py --checkpoint runs/<run>/best.pt
    python scripts/demo.py --no-browser --port 8001

Listens on 127.0.0.1 only: nothing is reachable from other computers.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # makes api/ importable when run as a script


def newest_checkpoint(runs: Path = Path("runs")) -> Path | None:
    """The most recently written best.pt (or model.safetensors) under runs/ or release/."""
    found = [*runs.glob("*/best.pt"), *Path("release").glob("*/model.safetensors")]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    ckpt = args.checkpoint or newest_checkpoint()
    if ckpt is None or not ckpt.exists():
        print(
            "No trained model found under runs/. Train one first, for example the preview "
            "model: python -m deepfake_detector.preview all",
            file=sys.stderr,
        )
        return 1
    os.environ["DFD_CHECKPOINT"] = str(ckpt)
    url = f"http://127.0.0.1:{args.port}"
    print(f"model: {ckpt}\ndemo:  {url}")

    import uvicorn
    from api.server import create_app

    from deepfake_detector.predict import Predictor

    app = create_app(Predictor(ckpt))  # load before serving so the first upload is quick
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
