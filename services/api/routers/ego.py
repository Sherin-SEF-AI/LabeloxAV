"""Ego trajectory and pseudo-3D coverage: build a session's pose, see how far coverage has got.

Reading is an annotator's business. Building a pose runs feature matching over a session's frames, which
is minutes of CPU, so it is a reviewer action; the nightly hook does the same work unasked.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import EgoPose
from services.api.deps import db_session, require_role

router = APIRouter()


@router.get("/ego/coverage", dependencies=[Depends(require_role("annotator"))])
async def ego_coverage(db: AsyncSession = Depends(db_session)):
    """Clouds, poses and 3D boxes against the real corpus. The number the nightly lift exists to move."""
    from services.lidar.pseudo_daemon import coverage

    return await coverage(db)


@router.get("/ego/sessions/{session_id}/pose", dependencies=[Depends(require_role("annotator"))])
async def session_pose(session_id: uuid.UUID, limit: int = Query(default=5000, ge=1, le=50000),
                       db: AsyncSession = Depends(db_session)):
    """One session's trajectory, for the viewer's overlay. Empty means nobody has built it yet."""
    rows = (await db.execute(select(EgoPose).where(EgoPose.session_id == session_id)
                             .order_by(EgoPose.ts_ns).limit(limit))).scalars().all()
    return {"session_id": str(session_id), "poses": len(rows),
            "measured": sum(1 for r in rows if r.measured),
            "source": rows[0].source if rows else None,
            "points": [{"ts_ns": r.ts_ns, "frame_id": str(r.frame_id) if r.frame_id else None,
                        "x": r.x, "y": r.y, "z": r.z, "qw": r.qw, "qz": r.qz,
                        "speed_mps": r.speed_mps, "yaw_rate": r.yaw_rate,
                        "quality": r.quality, "measured": r.measured} for r in rows]}


class PoseBuildIn(BaseModel):
    cam_id: str | None = None
    limit: int | None = Field(default=None, ge=2, le=20000)


@router.post("/ego/sessions/{session_id}/pose", dependencies=[Depends(require_role("reviewer"))])
async def build_pose(session_id: uuid.UUID, body: PoseBuildIn | None = None,
                     db: AsyncSession = Depends(db_session)):
    """Recover this session's trajectory from GNSS where it exists and from the images where it does not."""
    from services.intelligence.ego_pose import build_ego_pose

    body = body or PoseBuildIn()
    res = await build_ego_pose(session_id, cam_id=body.cam_id, limit=body.limit)
    if res.get("error"):
        raise HTTPException(status_code=404, detail=res["error"])
    if not res.get("poses"):
        raise HTTPException(status_code=409, detail=res.get("reason", "no pose could be recovered"))
    return res


@router.post("/ego/tracks/{track_id}/lift3d", dependencies=[Depends(require_role("reviewer"))])
async def lift_track_3d(track_id: uuid.UUID, dry_run: bool = Query(default=False),
                        db: AsyncSession = Depends(db_session)):
    """Place one track's cuboids in the session frame, lock their size, and smooth the trajectory."""
    from services.lidar.track3d.from2d import lift_track

    res = await lift_track(track_id, dry_run=dry_run)
    if res.get("reason") and not res.get("cuboids"):
        raise HTTPException(status_code=409, detail=res["reason"])
    return res
