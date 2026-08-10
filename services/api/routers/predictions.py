"""The immutable prediction plane: reads, plus ingestion of runs this engine did not itself perform.

The evaluation patch grid renders a false or true positive by cropping the prediction's bbox from its frame,
the mirror of GET /api/objects/{id}/crop for gold objects.

Predictions are append-only. `db/models.py` states that no path outside the inference writer may update or
delete one, and the ingestion route here honours that: it only ever inserts, and a caller resubmitting the
same work receives a new run id rather than mutating an old one, so previously published numbers stay
reproducible.
"""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from db.models import Frame, Prediction
from services.api.deps import db_session, require_role

# Evaluation is a reviewer/admin activity and this exposes frame imagery, so the floor is reviewer.
router = APIRouter(dependencies=[Depends(require_role("reviewer"))])


class ExternalModelIn(BaseModel):
    model_version: str
    task: str = "detection"
    notes: str | None = None


class ExternalRunIn(BaseModel):
    model_version: str
    # [{frame_id, class_name | class_id, bbox: [x1,y1,x2,y2], conf?, track_id?, rot_deg?, cuboid_3d?}]
    predictions: list[dict]
    gold_id: str | None = None
    # The caller's own code identity. Part of the reproducibility key, so two submissions from different
    # revisions of a customer's model are distinguishable rather than silently merged.
    code_sha: str = "external"
    params: dict | None = None


@router.post("/models/external")
async def register_external(payload: ExternalModelIn, db: AsyncSession = Depends(db_session)):
    """Register a model whose weights live outside this system, so its runs are attributable.

    There was no way to register a model at all: the registry is written by the training path, so a model
    trained anywhere else could not be named, and a run that cannot name its model produces a number nobody
    can act on.
    """
    from services.verdyx.external_run import register_external_model

    return await register_external_model(db, model_version=payload.model_version,
                                         task=payload.task, notes=payload.notes)


@router.post("/inference-runs")
async def ingest_run(payload: ExternalRunIn, db: AsyncSession = Depends(db_session)):
    """Ingest predictions from a model this engine did not run.

    Turns any model into something the harness can evaluate, the champion gate can compare against, and
    failure mining can consume, without its weights ever leaving the caller's network.
    """
    from services.verdyx.external_run import ingest_external_run

    try:
        return await ingest_external_run(
            db, model_version=payload.model_version, predictions=payload.predictions,
            gold_id=payload.gold_id, code_sha=payload.code_sha, params=payload.params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
