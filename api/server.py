"""FastAPI app and local demo page (FR-I3 to FR-I6).

Runs on localhost only. Each upload is written to a private temporary file, scored, and deleted
straight after, whatever happens (FR-I5). Nothing about uploads is stored or logged: no file
names, no contents, no scores. The one exception is opt-in: "Teach the model" in the page sends
that one file to /feedback, which keeps it under data/feedback/ on this computer for
python -m deepfake_detector.teach.

    python scripts/demo.py                       # newest trained model, opens the browser
    python scripts/demo.py --checkpoint runs/<run>/best.pt

or set DFD_CHECKPOINT and run: uvicorn api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from deepfake_detector import __version__
from deepfake_detector.predict import NOT_PROOF_NOTE, Predictor, png_base64

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", *IMAGES}
MAX_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
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
            "model_note": p.model_note if p else None,
            "note": NOT_PROOF_NOTE,
        }

    async def with_upload(
        file: UploadFile, allowed: set[str], limit: int, work: Callable[[str], Any]
    ) -> Any:
        """Write the upload to a private temp file, run `work` on it, always delete it."""
        _check_suffix(file, allowed)
        suffix = Path(file.filename or "").suffix.lower()
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="dfd_")
        try:
            size = 0
            with os.fdopen(fd, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File too large (limit {limit >> 20} MB).")
                    f.write(chunk)
            return await run_in_threadpool(work, tmp)
        finally:
            await file.close()
            Path(tmp).unlink(missing_ok=True)  # FR-I5: never kept

    @app.post("/predict")
    async def predict(file: UploadFile) -> dict[str, Any]:
        _check_suffix(file, ALLOWED)  # a wrong type is reported before a missing model
        predictor = get_predictor()
        result = await with_upload(file, ALLOWED, MAX_BYTES, predictor.predict)
        out: dict[str, Any] = result.to_dict()
        out["model"] = predictor.checkpoint_name
        out["calibrated"] = predictor.calibrated
        out["model_note"] = predictor.model_note
        return out

    @app.post("/test-fake")
    async def test_fake(file: UploadFile) -> dict[str, Any]:
        """Self-blend the face in a photo so there is a synthetic fake to try the detector on.

        The face keeps its identity; only a changed copy of itself is pasted back with a soft
        seam. Both images go back to the browser and nothing is kept here.
        """
        _check_suffix(file, IMAGES)  # a wrong type is reported before a missing model
        predictor = get_predictor()
        pair = await with_upload(file, IMAGES, MAX_IMAGE_BYTES, predictor.make_test_fake)
        if pair is None:
            raise HTTPException(422, "No face found in this photo.")
        original, blended = pair
        return {
            "original_png_base64": png_base64(original),
            "blended_png_base64": png_base64(blended),
            "note": "Synthetic test image: your face blended with a slightly changed copy of "
            "itself. Not stored.",
        }

    @app.post("/feedback")
    async def feedback(file: UploadFile, label: str = Form(...)) -> dict[str, Any]:
        """Keep one file the user marked as real or fake, for teaching the model later.

        Only this endpoint stores anything, and only when the user asks for it.
        """
        from deepfake_detector.teach import FEEDBACK_ROOT, example_counts, save_example

        _check_suffix(file, ALLOWED)
        if label not in ("real", "fake"):
            raise HTTPException(422, "label must be real or fake")
        data = await file.read(MAX_BYTES + 1)
        await file.close()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"File too large (limit {MAX_BYTES >> 20} MB).")
        suffix = Path(file.filename or "").suffix.lower()
        root = Path(os.environ.get("DFD_FEEDBACK_DIR", FEEDBACK_ROOT))
        await run_in_threadpool(save_example, data, suffix, label, root)
        return {"saved": True, "counts": example_counts(root)}

    return app


def _check_suffix(file: UploadFile, allowed: set[str]) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported file type. Use one of: {sorted(allowed)}")


app = create_app()
