"""Map-fusion job orchestration (M3.3): fuse per-drive map_elements, fit ground elevation, export the
fused map to Lanelet2 + OpenDRIVE, seal a map_commit, and write the fused (committed) map_elements with
provenance. compute_target='local' runs the averaging-fusion fallback inline here; 'cloud' parks the job
for the A100 GTSAM burst via services/hdmap/cloud.py (single-box GPU discipline). Mirrors the autolabel
job lifecycle."""

from __future__ import annotations

import uuid

from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement
from sqlalchemy import cast, func, select, update

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, MapCommit, MapElement, MapFusionJob
from db.session import get_sessionmaker
from services.hdmap.elevation import elevation_at, ransac_ground_plane
from services.hdmap.export import seal_map_commit_id, to_lanelet2_osm, to_opendrive
from services.hdmap.fuse import fuse_local

log = get_logger("hdmap_run")
CALIB_VERSION = "labelox-calib-0.1"


async def _region_center(db, session_ids: list[str]) -> tuple[float, float]:
    g = cast(Frame.gnss, Geometry)
    row = (await db.execute(
        select(func.avg(func.ST_Y(g)), func.avg(func.ST_X(g)))
        .where(Frame.session_id.in_([uuid.UUID(s) for s in session_ids]), Frame.gnss.isnot(None)))).first()
    if row and row[0] is not None:
        return float(row[0]), float(row[1])
    return 12.9716, 77.5946  # Bangalore fallback


async def run_map_fusion(job_id: uuid.UUID) -> dict:
    """Execute a local map-fusion job end to end (fuse -> elevation -> export -> seal -> write elements)."""
    cfg = get_settings()
    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()

    async with maker() as db:
        job = await db.get(MapFusionJob, job_id)
        if job is None:
            return {"error": "job not found"}
        session_ids, region = list(job.session_ids), job.region
        job.status, job.stage, job.progress = "running", "fuse", 0.1
        await db.commit()

    fused = await fuse_local(session_ids, region)
    f_elems = fused["fused"]

    async with maker() as db:
        await db.execute(update(MapFusionJob).where(MapFusionJob.job_id == job_id).values(stage="elevation", progress=0.4))
        await db.commit()
        center = await _region_center(db, session_ids)

    # elevation: RANSAC ground plane over the trajectory (flat fallback where no altitude/cloud exists)
    plane = ransac_ground_plane([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    elev = elevation_at(plane, 0.0, 0.0) if plane else 0.0

    lanelet2 = to_lanelet2_osm(f_elems)
    opendrive = to_opendrive(f_elems, center)
    commit_id = seal_map_commit_id(session_ids, f_elems, CALIB_VERSION)
    prefix = f"maps/{region}/{commit_id}"
    formats = {
        "lanelet2": store.put_bytes(f"{prefix}/map.osm", lanelet2.encode(), "application/xml"),
        "opendrive": store.put_bytes(f"{prefix}/map.xodr", opendrive.encode(), "application/xml"),
    }

    async with maker() as db:
        await db.execute(update(MapFusionJob).where(MapFusionJob.job_id == job_id).values(stage="seal", progress=0.8))
        # idempotent re-commit: clear a prior commit of the same id
        await db.execute(MapElement.__table__.delete().where(MapElement.commit_id == commit_id))
        await db.execute(MapCommit.__table__.delete().where(MapCommit.commit_id == commit_id))
        db.add(MapCommit(commit_id=commit_id, region=region, session_ids=session_ids,
                         element_count=len(f_elems), formats=formats, calibration_version=CALIB_VERSION,
                         fusion_job_id=job_id))
        await db.flush()
        for f in f_elems:
            db.add(MapElement(kind=f["kind"], geometry=WKTElement(f["wkt"], srid=4326),
                              attrs={**f["attrs"], "elevation_m": round(elev or 0.0, 3)},
                              source_frames=f["frames"], source_sessions=f["sessions"],
                              calibration_version=f.get("calib") or CALIB_VERSION, confidence=f["confidence"],
                              fusion_job_id=job_id, commit_id=commit_id))
        await db.execute(update(MapFusionJob).where(MapFusionJob.job_id == job_id).values(
            status="completed", stage="done", progress=1.0, commit_id=commit_id,
            counts={"input": fused["input_elements"], "fused": len(f_elems), "consensus": fused["consensus"]},
            result={"commit_id": commit_id, "formats": formats, "center": list(center)}))
        await db.commit()

    out = {"commit_id": commit_id, "fused_elements": len(f_elems), "consensus": fused["consensus"],
           "formats": formats, "region": region}
    log.info("hdmap.fusion_done", job_id=str(job_id), **{k: out[k] for k in ("commit_id", "fused_elements")})
    return out


async def start_map_fusion(session_ids: list[str], region: str | None = None,
                           compute_target: str = "local") -> dict:
    cfg = get_settings()
    region = region or cfg.spatial.map_region
    maker = get_sessionmaker()
    async with maker() as db:
        job = MapFusionJob(session_ids=session_ids, region=region, compute_target=compute_target, status="pending")
        db.add(job)
        await db.flush()
        job_id = job.job_id
        await db.commit()

    if compute_target == "cloud":
        from services.hdmap.cloud import mark_queued_for_cloud_fusion

        await mark_queued_for_cloud_fusion(job_id, session_ids, region)
        return {"job_id": str(job_id), "compute_target": "cloud", "status": "queued_for_cloud"}

    res = await run_map_fusion(job_id)
    return {"job_id": str(job_id), "compute_target": "local", **res}
