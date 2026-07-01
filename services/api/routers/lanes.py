"""Lane annotation endpoints (M2.1): propose (CLRerNet on pod / classical local), list, create/update/
delete human lanes, and propagate a frame's lanes forward by optical flow (source=propagated)."""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from db.models import Frame, Lane
from services.api.deps import db_session
from services.autolabel.lane.curves import fit_control_points, mark_ego, propagate_control_points
from services.autolabel.lane.detect import model_tag, propose_lanes

router = APIRouter()


class LaneIn(BaseModel):
    control_points: list
    lane_type: str = "solid"
    is_ego: bool = False


def _decode(store, uri):
    return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)


def _row(lane: Lane) -> dict:
    return {"lane_id": str(lane.lane_id), "frame_id": str(lane.frame_id),
            "track_ref": str(lane.track_ref) if lane.track_ref else None,
            "control_points": lane.control_points, "lane_type": lane.lane_type,
            "is_ego": lane.is_ego, "source": lane.source, "model_version": lane.model_version}


@router.get("/frames/{frame_id}/lanes")
async def list_lanes(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(Lane).where(Lane.frame_id == frame_id))).scalars().all()
    return [_row(lane) for lane in rows]


@router.post("/frames/{frame_id}/lanes/propose")
async def propose(frame_id: UUID, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    img = _decode(get_object_store(), frame.img_uri)
    if img is None:
        raise HTTPException(500, "could not decode frame image")
    await db.execute(delete(Lane).where(Lane.frame_id == frame.frame_id, Lane.source == "proposed"))
    cps = [fit_control_points(p) for p in propose_lanes(img)]
    ego = mark_ego(cps, frame.width, frame.height)
    created = []
    for i, cp in enumerate(cps):
        lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=cp,
                    lane_type="solid", is_ego=(i == ego), source="proposed", model_version=model_tag())
        db.add(lane)
        created.append(lane)
    await db.flush()
    out = [_row(lane) for lane in created]
    await db.commit()
    return {"proposed": len(out), "lanes": out, "model": model_tag()}


@router.post("/frames/{frame_id}/lanes")
async def create_lane(frame_id: UUID, body: LaneIn, db: AsyncSession = Depends(db_session)):
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=body.control_points,
                lane_type=body.lane_type, is_ego=body.is_ego, source="human")
    db.add(lane)
    await db.flush()
    r = _row(lane)
    await db.commit()
    return r


@router.put("/lanes/{lane_id}")
async def update_lane(lane_id: UUID, body: LaneIn, db: AsyncSession = Depends(db_session)):
    lane = await db.get(Lane, lane_id)
    if lane is None:
        raise HTTPException(404, "lane not found")
    lane.control_points, lane.lane_type, lane.is_ego, lane.source = body.control_points, body.lane_type, body.is_ego, "human"
    lane.model_version = None  # a human now owns this lane; drop the stale proposing-model tag so provenance is not "human - clrernet"
    await db.commit()
    return _row(lane)


@router.delete("/lanes/{lane_id}")
async def delete_lane(lane_id: UUID, db: AsyncSession = Depends(db_session)):
    lane = await db.get(Lane, lane_id)
    if lane is None:
        raise HTTPException(404, "lane not found")
    await db.delete(lane)
    await db.commit()
    return {"deleted": str(lane_id)}


@router.post("/frames/{frame_id}/lanes/propagate")
async def propagate(frame_id: UUID, frames: int = 8, db: AsyncSession = Depends(db_session)):
    """Carry this frame's lanes forward via optical flow; the annotator only fixes keyframes."""
    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    store = get_object_store()
    lanes = (await db.execute(select(Lane).where(Lane.frame_id == frame.frame_id))).scalars().all()
    if not lanes:
        return {"created": 0, "reason": "no lanes on the source frame"}
    nexts = (await db.execute(
        select(Frame).where(Frame.session_id == frame.session_id, Frame.ts_ns > frame.ts_ns)
        .order_by(Frame.ts_ns).limit(frames))).scalars().all()
    prev_gray = cv2.cvtColor(_decode(store, frame.img_uri), cv2.COLOR_BGR2GRAY)
    cur = {lane.lane_id: lane.control_points for lane in lanes}
    meta = {lane.lane_id: (lane.lane_type, lane.is_ego, lane.track_ref or lane.lane_id) for lane in lanes}
    created = 0
    for nf in nexts:
        nimg = _decode(store, nf.img_uri)
        if nimg is None:
            break
        cur_gray = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY)
        for lid in list(cur):
            ncp = propagate_control_points(prev_gray, cur_gray, cur[lid])
            if ncp is None:
                continue
            lt, ego, ref = meta[lid]
            db.add(Lane(frame_id=nf.frame_id, session_id=frame.session_id, control_points=ncp,
                        lane_type=lt, is_ego=ego, source="propagated", track_ref=ref, model_version="optical-flow"))
            cur[lid] = ncp
            created += 1
        prev_gray = cur_gray
    await db.commit()
    return {"created": created, "to_frames": len(nexts)}
