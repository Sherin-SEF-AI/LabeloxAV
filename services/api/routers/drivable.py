"""Drivable-surface endpoints (M2.2): segment a frame into a ternary surface mask (SAM 3.1 PCS on pod /
sam_b road-seed local), store polygons-per-class in MinIO + coverage on drivable_mask, fetch, and refine."""

from __future__ import annotations

import json
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from db.models import DrivableMask, Frame
from services.api.deps import current_user, db_session, require_role
from services.autolabel.drivable import segment_drivable

router = APIRouter()


class DrivableIn(BaseModel):
    classes: dict  # {drivable: [[x,y,...]], non_drivable: [...], fallback: [...]}
    width: int
    height: int


def _decode(store, uri):
    return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)


def _coverage(classes: dict, w: int, h: int) -> dict:
    out = {}
    total = float(w * h) or 1.0
    for name, polys in classes.items():
        m = np.zeros((h, w), np.uint8)
        for p in polys:
            pts = np.asarray(p, np.float32).reshape(-1, 2).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
        out[name] = round(float(m.sum()) / total, 4)
    return out


async def _store_mask(db, frame, classes, width, height, coverage, model, source):
    store = get_object_store()
    key = f"masks/drivable/{frame.session_id}/{frame.frame_id}.json"
    uri = store.put_bytes(key, json.dumps({"classes": classes, "width": width, "height": height}).encode(), "application/json")
    dm = await db.get(DrivableMask, frame.frame_id)
    if dm is None:
        db.add(DrivableMask(frame_id=frame.frame_id, mask_uri=uri, coverage=coverage, source=source, model_version=model))
    else:
        dm.mask_uri, dm.coverage, dm.model_version, dm.source = uri, coverage, model, source
    await db.commit()
    return uri


# Annotator floor, matching POST /frames/{id}/segment (services/api/routers/segmentation.py:31). This was
# the only GPU-spending POST route in the tree without one: any authenticated caller could trigger a
# full-resolution 215M-parameter forward pass, one per request, with nothing rate-limiting it.
@router.post("/frames/{frame_id}/drivable", dependencies=[Depends(require_role("annotator"))])
async def segment_frame(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    img = _decode(get_object_store(), frame.img_uri)
    if img is None:
        raise HTTPException(500, "could not decode frame image")
    out = segment_drivable(img)
    uri = await _store_mask(db, frame, out["classes"], out["width"], out["height"], out["coverage"], out["model"], "proposed")
    # unestimated_classes tells the caller which surface classes this backend cannot measure, so a 0.0
    # coverage is not misread as "measured and absent" (the geometric fallback cannot see unpaved surface).
    return {"frame_id": str(frame_id), "coverage": out["coverage"], "mask_uri": uri, "model": out["model"],
            "unestimated_classes": out.get("unestimated_classes", [])}


@router.get("/frames/{frame_id}/drivable")
async def get_drivable(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    dm = await db.get(DrivableMask, frame_id)
    if dm is None:
        return {"found": False}
    data = json.loads(get_object_store().get_bytes(dm.mask_uri))
    return {"found": True, "coverage": dm.coverage, "source": dm.source, "model_version": dm.model_version,
            "classes": data["classes"], "width": data["width"], "height": data["height"]}


@router.put("/frames/{frame_id}/drivable")
async def refine_drivable(frame_id: UUID, body: DrivableIn, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    cov = _coverage(body.classes, body.width, body.height)
    uri = await _store_mask(db, frame, body.classes, body.width, body.height, cov, "human", "human")
    return {"frame_id": str(frame_id), "coverage": cov, "mask_uri": uri}


# Corpus backfill. The single-frame POST above is how drivable was produced until now, which is why 96%
# of the corpus has none: it only ran when somebody opened a frame and clicked.


@router.get("/drivable/coverage")
async def drivable_coverage(db: AsyncSession = Depends(db_session)):
    """How much of the corpus carries a mask, by capture size.

    Broken down because the headline hides the shape: the buckets that look finished are small older
    imports, and the main dashcam corpus is the empty one.
    """
    from services.perception.backfill import coverage

    return await coverage(db)


@router.post("/drivable/backfill", dependencies=[Depends(require_role("reviewer"))])
async def drivable_backfill(session_id: str | None = None, max_frames: int = 50_000,
                            redo: bool = False, db: AsyncSession = Depends(db_session),
                            user=Depends(current_user)):
    """Segment every frame that has no mask. Reviewer floor: it holds the GPU for as long as it runs."""
    from core.observability import spawn
    from services.perception.backfill import run_drivable_backfill, start_backfill

    res = await start_backfill(db, session_id=session_id, max_frames=max_frames, redo=redo,
                               created_by=str(user.user_id) if user else None)
    spawn(run_drivable_backfill(UUID(res["run_id"]), session_id=session_id,
                                max_frames=max_frames, redo=redo), name="drivable_backfill")
    return res
