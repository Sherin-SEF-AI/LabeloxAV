"""OCR endpoints (M2.4): read road text (PaddleOCR Indic on pod / Qwen local), excluding license plates."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.api.deps import require_role

router = APIRouter()


@router.post("/ocr/run")
async def run(session_id: str, limit: int | None = None):
    """OCR a session's text-bearing sign/board objects; plate-overlapping regions are excluded."""
    from services.autolabel.ocr.reader import ocr_session

    return await ocr_session(UUID(session_id), limit)


class RegionOcrIn(BaseModel):
    # An asset or object-store uri, and the region to read in image pixels.
    uri: str
    bbox: list[float]
    pad: float = 0.02


@router.post("/ocr/region", dependencies=[Depends(require_role("annotator"))])
async def ocr_region(payload: RegionOcrIn) -> dict:
    """Read the text in one region of an image, for the document annotator's transcription field.

    Separate from /ocr/run, which walks a session's sign objects. A document page is an Asset, not a Frame,
    and its regions are drawn by hand rather than detected, so there is nothing for the session path to
    iterate. Without this the document editor could box a region and then required the annotator to retype
    what was already legible on screen.

    The confidence comes back as None from the local reader and is passed through as None. A number here
    would make the transcription look verified when nobody verified it, which is the mistake the OCR path
    was already fixed once for.
    """
    import cv2
    import numpy as np

    from core.storage import get_object_store
    from services.autolabel.ocr.reader import read_text

    if len(payload.bbox) != 4:
        raise HTTPException(400, "bbox must be [x1, y1, x2, y2] in image pixels")

    try:
        raw = get_object_store().get_bytes(payload.uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "could not read that image") from exc
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "that image could not be decoded")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = payload.bbox
    px, py = (x2 - x1) * payload.pad, (y2 - y1) * payload.pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "that region is empty")

    try:
        text, lang, conf = read_text(img[y1:y2, x1:x2])
    except NotImplementedError as exc:
        # The pod backend is selected and absent. Refused by name rather than returning an empty string,
        # which would read as "this region contains no text".
        raise HTTPException(503, str(exc)) from exc
    return {"text": text, "lang": lang, "conf": conf,
            "measured": conf is not None,
            "bbox": [x1, y1, x2, y2]}
