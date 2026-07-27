"""Read-only access to the immutable prediction plane. The evaluation patch grid renders a false/true positive
by cropping the prediction's bbox from its frame, the mirror of GET /api/objects/{id}/crop for gold objects.
Predictions are never mutated here (or anywhere outside the inference writer); this router only reads.
"""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from db.models import Frame, Prediction
from services.api.deps import db_session, require_role

# Evaluation is a reviewer/admin activity and this exposes frame imagery, so the floor is reviewer.
router = APIRouter(dependencies=[Depends(require_role("reviewer"))])


@router.get("/predictions/{prediction_id}/crop")
async def prediction_crop(prediction_id: str, pad: float = 0.15, db: AsyncSession = Depends(db_session)):
    """A JPEG crop of a prediction's bbox (with padding), for the evaluation patch grid."""
    pred = await db.get(Prediction, UUID(prediction_id))
    if pred is None:
        raise HTTPException(404, "prediction not found")
    frame = await db.get(Frame, pred.frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    try:
        buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "frame image unavailable") from exc
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(404, "failed to decode frame image")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = pred.bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = img
    _ok, out = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=out.tobytes(), media_type="image/jpeg")
