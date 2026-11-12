"""FastAPI app and local demo page (FR-I3 to FR-I6).

Runs on localhost only. Each upload is written to a private temporary file, scored, and deleted
straight after, whatever happens (FR-I5). Nothing about uploads is stored or logged: no file
names, no contents, no scores.

    set DFD_CHECKPOINT=runs\\<run>\\best.pt
    uvicorn api.server:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from deepfake_detector import __version__
from deepfake_detector.predict import NOT_PROOF_NOTE, Predictor

ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BYTES = 200 * 1024 * 1024
PAGE = (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="deepfake-detector", version=__version__, docs_url=None, redoc_url=None)
    state: dict[str, Any] = {"predictor": predictor}

    def get_predictor() -> Predictor:
        if state["predictor"] is None:
            ckpt = os.environ.get("DFD_CHECKPOINT")
            if not ckpt:
                raise HTTPException(503, "No model loaded. Set DFD_CHECKPOINT to a checkpoint.")
            state["predictor"] = Predictor(ckpt)
        return state["predictor"]  # type: ignore[no-any-return]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/health")
    def health() -> dict[str, Any]:
        p = state["predictor"]
        return {
            "status": "ok",
            "version": __version__,
            "model_loaded": p is not None,
            "model": p.checkpoint_name if p else None,
            "device": p.device.type if p else None,
            "calibrated": p.calibrated if p else None,
            "note": NOT_PROOF_NOTE,
        }

    @app.post("/predict")
    async def predict(file: UploadFile) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, f"Unsupported file type. Use one of: {sorted(ALLOWED)}")
        predictor = get_predictor()
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="dfd_")
        try:
            size = 0
            with os.fdopen(fd, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "File too large (limit 200 MB).")
                    f.write(chunk)
            result = await run_in_threadpool(predictor.predict, tmp)
        finally:
            await file.close()
            Path(tmp).unlink(missing_ok=True)  # FR-I5: never kept
        out = result.to_dict()
        out["model"] = predictor.checkpoint_name
        out["calibrated"] = predictor.calibrated
        return out

    return app


app = create_app()
