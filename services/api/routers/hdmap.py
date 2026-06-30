"""HD map endpoints (M3.3): geo-reference a session's annotations to world, fuse multi-drive maps (local
or A100 burst), serve fused elements as GeoJSON for the MapLibre viewer, list map commits, and walk an
element's provenance to its source frames + calibration + fusion run."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from geoalchemy2 import Geometry
from pydantic import BaseModel
from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, MapCommit, MapElement
from db.models import Session as DbSession
from services.api.deps import db_session, require_role

router = APIRouter()


@router.post("/hdmap/georef")
async def georef(session_id: str, height_m: float | None = None):
    from services.hdmap.georef import georef_session

    return await georef_session(UUID(session_id), height_m)


class MetricElementIn(BaseModel):
    kind: str                                 # lane|road_edge|stop_line|crosswalk|crossing
    points_ego: list                          # [[x_forward, y_left], ...] metres
    ref_lat: float
    ref_lon: float
    heading_rad: float = 0.0


@router.post("/hdmap/elements/metric", dependencies=[Depends(require_role("annotator"))])
async def create_metric_element_ep(session_id: str, body: MetricElementIn):
    """Milestone F: a 3D metric polyline (lane boundary, stop line, crosswalk) drawn in the ego/BEV frame,
    georeferenced into a world-space MapElement."""
    from fastapi import HTTPException

    from services.hdmap.metric_vector import create_metric_element
    res = await create_metric_element(UUID(session_id), body.kind, body.points_ego, body.ref_lat,
                                      body.ref_lon, body.heading_rad)
    if res.get("error"):
        raise HTTPException(422, res["error"])
    return res


@router.post("/hdmap/fuse")
async def fuse(session_ids: str = Query(..., description="comma-separated session ids"),
               region: str | None = None, compute_target: str = "local"):
    from services.hdmap.run import start_map_fusion

    ids = [s.strip() for s in session_ids.split(",") if s.strip()]
    return await start_map_fusion(ids, region, compute_target)


@router.get("/hdmap/commits")
async def commits(db: AsyncSession = Depends(db_session), limit: int = Query(200, ge=1, le=1000)):
    rows = (await db.execute(select(MapCommit).order_by(MapCommit.created_at.desc()).limit(limit))).scalars().all()
    return [{"commit_id": c.commit_id, "region": c.region, "element_count": c.element_count,
             "session_ids": c.session_ids, "formats": c.formats, "calibration_version": c.calibration_version,
             "created_at": c.created_at.isoformat() if c.created_at else None} for c in rows]


@router.get("/hdmap/elements")
async def elements(commit_id: str | None = None, session_id: str | None = None,
                   db: AsyncSession = Depends(db_session)):
    """GeoJSON FeatureCollection of map elements (by commit, or per-drive by session)."""
    g = cast(MapElement.geometry, Geometry)
    q = select(MapElement.element_id, MapElement.kind, MapElement.attrs, MapElement.confidence,
               MapElement.calibration_version, MapElement.commit_id, func.ST_AsGeoJSON(g))
    if commit_id:
        q = q.where(MapElement.commit_id == commit_id)
    elif session_id:
        q = q.where(MapElement.source_sessions.any(session_id))
    else:
        q = q.where(MapElement.commit_id.isnot(None))
    rows = (await db.execute(q)).all()
    features = [{
        "type": "Feature", "geometry": json.loads(gj) if gj else None,
        "properties": {"element_id": str(eid), "kind": kind, "confidence": round(conf or 0.0, 2),
                       "calibration_version": calib, "commit_id": cid, **(attrs or {})},
    } for eid, kind, attrs, conf, calib, cid, gj in rows]
    return {"type": "FeatureCollection", "features": features}


@router.get("/hdmap/provenance")
async def provenance(element_id: UUID, db: AsyncSession = Depends(db_session)):
    el = await db.get(MapElement, element_id)
    if el is None:
        return {"found": False}
    frames = []
    for fid in (el.source_frames or []):
        try:
            f = await db.get(Frame, UUID(fid))
        except Exception:  # noqa: BLE001
            f = None
        if f is None:
            continue
        sess = await db.get(DbSession, f.session_id)
        frames.append({"frame_id": fid, "session_id": str(f.session_id), "cam_id": f.cam_id, "ts_ns": f.ts_ns,
                       "vehicle_id": sess.vehicle_id if sess else None,
                       "ontology_version": sess.ontology_version if sess else None})
    return {"found": True, "element_id": str(element_id), "kind": el.kind, "attrs": el.attrs,
            "confidence": el.confidence, "calibration_version": el.calibration_version,
            "commit_id": el.commit_id, "fusion_job_id": str(el.fusion_job_id) if el.fusion_job_id else None,
            "source_sessions": el.source_sessions, "source_frames": frames}
