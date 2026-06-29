"""Interactive pixel-assist endpoints: brush/eraser mask composition and SLIC superpixels. The
magic-wand (SAM point) reuses the existing POST /segment; these two add the GPU-free assists.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame
from services.api.deps import db_session, require_role

router = APIRouter()


class ComposeIn(BaseModel):
    polygons: list[list[float]] = []   # the object's current polygons, flattened [x,y,...] each
    ops: list[dict] = []               # [{"op": "add"|"erase", "center": [x,y], "radius": r}, ...]
    width: int
    height: int


@router.post("/mask/compose", dependencies=[Depends(require_role("annotator"))])
async def compose(payload: ComposeIn):
    """Apply brush/eraser stamps to a mask and return the recomputed polygons."""
    from services.segment2d.assist import compose_mask
    return {"polygons": compose_mask(payload.polygons, payload.ops, payload.width, payload.height)}


@router.post("/superpixels/{frame_id}")
async def superpixels(frame_id: str, n: int = 300, db: AsyncSession = Depends(db_session)):
    """SLIC superpixels for a frame, as polygons to click into a mask."""
    from core.storage import get_object_store
    from services.recall.backends import load_image_bgr
    from services.segment2d.assist import slic_superpixels

    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    try:
        img = load_image_bgr(get_object_store(), frame.img_uri)
    except Exception as exc:  # noqa: BLE001  (a missing frame blob must not 500 the editor)
        raise HTTPException(404, "frame image unavailable") from exc
    return {"superpixels": slic_superpixels(img, n)}
