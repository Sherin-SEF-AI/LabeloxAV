"""Orchestrate multi-scan aggregation across sessions: register consecutive scans with the GNSS/IMU prior,
chain the poses, detect and close loops, accumulate into a dense map, and write an aggregated_map row. The
mean registration fitness is recorded, and a low mean flags low-confidence pseudo-LiDAR registration.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import AggregatedMap, Frame, PointCloud
from db.session import get_sessionmaker
from services.lidar.aggregate.accumulate import accumulate_scans
from services.lidar.aggregate.loopclose import detect_loops, optimize_pose_graph
from services.lidar.aggregate.register import accumulate_poses, gnss_imu_prior, register_pair
from services.lidar.ingest.store import load_cloud

log = get_logger("lidar_aggregate")
METHOD = "register-loopclose-accumulate-0.1"


async def _session_clouds(session_id: uuid.UUID) -> list[dict]:
    async with get_sessionmaker()() as db:
        geom = cast(Frame.gnss, Geometry)
        gnss = dict(((int(ts)), (float(la), float(lo))) for ts, la, lo in (await db.execute(
            select(Frame.ts_ns, func.ST_Y(geom), func.ST_X(geom))
            .where(Frame.session_id == session_id, Frame.gnss.isnot(None)))).all())
        rows = (await db.execute(select(PointCloud).where(PointCloud.session_id == session_id)
                .order_by(PointCloud.ts_ns))).scalars().all()
    out = []
    for r in rows:
        latlon = gnss.get(int(r.ts_ns))
        out.append({"cloud_id": r.cloud_id, "ts_ns": int(r.ts_ns), "uri": r.cloud_uri, "latlon": latlon})
    return out


def _prior(a: dict, b: dict) -> np.ndarray:
    if not a.get("latlon") or not b.get("latlon"):
        return np.eye(4)
    (la0, lo0), (la1, lo1) = a["latlon"], b["latlon"]
    north = (la1 - la0) * 111320.0
    east = (lo1 - lo0) * 111320.0 * math.cos(math.radians(la0))
    return gnss_imu_prior(east, north, 0.0)


async def aggregate_sessions(session_ids: list[uuid.UUID], region: str | None = None,
                             voxel: float = 0.2) -> dict:
    """Register, loop-close, and accumulate the clouds of one or more sessions into a dense aggregated map."""
    metas: list[dict] = []
    for sid in session_ids:
        metas += await _session_clouds(sid)
    metas.sort(key=lambda m: m["ts_ns"])
    if len(metas) < 2:
        return {"region": region, "n_scans": len(metas), "reason": "need at least two scans"}

    clouds = [load_cloud(m["uri"]) for m in metas]
    transforms, fitnesses = [], []
    for i in range(len(clouds) - 1):
        reg = register_pair(clouds[i + 1].xyz, clouds[i].xyz, init=_prior(metas[i], metas[i + 1]))
        transforms.append(np.asarray(reg["transformation"]))
        fitnesses.append(reg["fitness"])

    poses = accumulate_poses(transforms)
    loops = detect_loops(poses)
    pg = optimize_pose_graph(poses, loops) if loops else {"method": "none", "loops": []}
    agg = accumulate_scans(clouds, poses, voxel=voxel)

    store = get_object_store()
    store.ensure_bucket()
    cfg = get_settings().lidar
    cloud_uri = store.put_bytes(f"{cfg.cloud_prefix}/aggregated/{uuid.uuid4()}.npz", agg.to_npz_bytes(),
                                "application/octet-stream")
    mean_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
    low_conf = mean_fitness < cfg.register_min_fitness

    async with get_sessionmaker()() as db:
        row = AggregatedMap(region=region, session_ids=list(session_ids), cloud_uri=cloud_uri,
                            pose_graph={"poses": [p.tolist() for p in poses], "fitnesses": fitnesses},
                            loop_closures=pg, method=METHOD, n_scans=len(clouds),
                            mean_reg_fitness=round(mean_fitness, 4))
        db.add(row)
        await db.flush()
        agg_id = row.agg_id
        await db.commit()
    log.info("lidar.aggregate", region=region, scans=len(clouds), loops=len(loops),
             mean_fitness=round(mean_fitness, 3), low_conf=low_conf, points=agg.n)
    return {"agg_id": str(agg_id), "region": region, "n_scans": len(clouds), "points": agg.n,
            "loops": len(loops), "mean_reg_fitness": round(mean_fitness, 4),
            "low_confidence_registration": low_conf, "cloud_uri": cloud_uri}
