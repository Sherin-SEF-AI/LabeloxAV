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


class OccupancyBuildIn(BaseModel):
    t0: int | None = None
    t1: int | None = None
    limit: int = Field(default=200, ge=1, le=5000)
    voxel_m: float | None = Field(default=None, gt=0.05, le=5.0)


@router.post("/occupancy/sessions/{session_id}/build",
             dependencies=[Depends(require_role("reviewer"))])
async def build_occupancy(session_id: uuid.UUID, body: OccupancyBuildIn | None = None,
                          db: AsyncSession = Depends(db_session)):
    """Turn this session's point clouds into occupancy grids with a flow field, batch by batch."""
    from services.lidar.occupancy4d import VOXEL_M, build_occupancy_window

    body = body or OccupancyBuildIn()
    res = await build_occupancy_window(session_id, body.t0, body.t1, limit=body.limit,
                                       voxel_m=body.voxel_m or VOXEL_M)
    if not res.get("grids"):
        raise HTTPException(status_code=409, detail=res.get("reason", "no grid could be built"))
    return res


@router.get("/occupancy/sessions/{session_id}", dependencies=[Depends(require_role("annotator"))])
async def list_occupancy(session_id: uuid.UUID, limit: int = Query(default=500, ge=1, le=5000),
                         db: AsyncSession = Depends(db_session)):
    """This session's grids in time order, for the viewer's scrubber.

    `flow_share` is on every row rather than left to the caller to divide, because it is the number that
    says whether a grid's velocity field is measured or assumed, and a viewer drawing arrows from an
    assumed-zero field would be drawing confidence nobody has.
    """
    from db.models import OccupancyGrid

    rows = (await db.execute(select(OccupancyGrid).where(OccupancyGrid.session_id == session_id)
                             .order_by(OccupancyGrid.ts_ns).limit(limit))).scalars().all()
    return {"session_id": str(session_id), "n": len(rows),
            "grids": [{"grid_id": str(g.grid_id), "ts_ns": g.ts_ns,
                       "frame_id": str(g.frame_id) if g.frame_id else None,
                       "origin": [float(v) for v in g.origin], "voxel_m": g.voxel_m,
                       "dims": [int(v) for v in g.dims], "source": g.source,
                       "occupied": g.occupied, "flow_voxels": g.flow_voxels,
                       "flow_share": (round(g.flow_voxels / g.occupied, 4) if g.occupied else None),
                       "placed_by_pose": g.ego_pose_ts is not None} for g in rows]}


@router.get("/occupancy/{grid_id}/voxels", dependencies=[Depends(require_role("annotator"))])
async def occupancy_voxels(grid_id: uuid.UUID, max_voxels: int = Query(default=60000, ge=100,
                                                                       le=500000),
                           db: AsyncSession = Depends(db_session)):
    """The unpacked voxels and their flow, bounded, for drawing.

    Bounded because a dense urban grid is tens of thousands of voxels and a browser asked to draw all of
    them stops being a viewer. When it truncates it says so, so a thin-looking scene reads as a display
    limit rather than as empty road.
    """
    from core.storage import get_object_store
    from db.models import OccupancyGrid
    from services.lidar.occupancy4d import unpack_grid

    g = await db.get(OccupancyGrid, grid_id)
    if g is None:
        raise HTTPException(status_code=404, detail="grid not found")
    try:
        data = unpack_grid(get_object_store().get_bytes(g.grid_uri))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"the packed grid could not be read: {exc}") from exc

    voxels, flow = data["voxels"], data["flow"]
    truncated = len(voxels) > max_voxels
    if truncated:
        voxels, flow = voxels[:max_voxels], flow[:max_voxels]
    return {"grid_id": str(grid_id), "ts_ns": g.ts_ns, "voxel_m": g.voxel_m,
            "origin": [float(v) for v in g.origin], "dims": [int(v) for v in g.dims],
            "n": int(len(voxels)), "total": int(g.occupied), "truncated": truncated,
            "voxels": voxels.tolist(), "flow": [[round(float(v), 3) for v in f] for f in flow],
            "placed_by_pose": g.ego_pose_ts is not None}
