"""FastAPI app and local demo page (FR-I3 to FR-I6).

Runs on localhost only. Uploads are deleted after each prediction and nothing about them is
stored or logged (FR-I5).

Run: uvicorn api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from deepfake_detector import __version__
from deepfake_detector.predict import NOT_PROOF_NOTE

app = FastAPI(title="deepfake-detector", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "note": NOT_PROOF_NOTE}
