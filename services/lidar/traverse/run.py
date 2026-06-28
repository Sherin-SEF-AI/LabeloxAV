"""Orchestrate 3D traversability for a cloud: free-space grid, metric drivable grid, road-surface class, and
elevation profile. Grids are written to the object store; the surface and elevation summaries are inline. One
traversability row per cloud.
"""

from __future__ import annotations

import io
import uuid

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Traversability
from db.session import get_sessionmaker
from services.lidar.extract.common import load_for_extraction
from services.lidar.segment3d.semantic import road_class_id
from services.lidar.traverse.drivable3d import drivable_grid
from services.lidar.traverse.elevation import elevation_profile
from services.lidar.traverse.freespace import freespace_grid
from services.lidar.traverse.surface import classify_surface

log = get_logger("lidar_traverse")
METHOD = "occupancy-grid-0.1"


def _store_grid(store, session_id: uuid.UUID, cloud_id: uuid.UUID, name: str, grid: dict) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, grid=grid["grid"], res=grid["res"], x_range=grid["x_range"],
                        y_range=grid["y_range"])
    cfg = get_settings().lidar
    return store.put_bytes(f"{cfg.cloud_prefix}/{session_id}/traverse/{cloud_id}_{name}.npz",
                           buf.getvalue(), "application/octet-stream")


async def traverse_cloud(cloud_id: uuid.UUID) -> dict:
    """Produce and store the free-space and drivable grids, the surface class, and the elevation profile."""
    data = await load_for_extraction(cloud_id)
    if data is None:
        return {"error": "cloud not found"}
    cloud, plane, semantic, session_id = data["cloud"], data["plane"], data["semantic"], data["session_id"]
    calib = data.get("calibration_version")
    road_id = road_class_id()

    fs = freespace_grid(cloud, plane)
    dr = drivable_grid(cloud, semantic, road_id, plane)
    surf = classify_surface(cloud, semantic, road_id, plane)
    elev = elevation_profile(cloud, plane)

    store = get_object_store()
    store.ensure_bucket()
    fs_uri = _store_grid(store, session_id, cloud_id, "freespace", fs)
    dr_uri = _store_grid(store, session_id, cloud_id, "drivable", dr)

    async with get_sessionmaker()() as db:
        row = Traversability(cloud_id=cloud_id, freespace_uri=fs_uri, drivable_uri=dr_uri,
                             surface_class=surf, method=METHOD, calibration_version=calib,
                             elevation_profile={k: v for k, v in elev.items() if k != "profile"} | {
                                 "profile": elev["profile"]})
        db.add(row)
        await db.flush()
        tid = row.id
        await db.commit()
    log.info("lidar.traverse_cloud", cloud=str(cloud_id), free_frac=fs["free_frac"],
             drivable_frac=dr["drivable_frac"], surface=surf["surface"], feature=elev["feature"])
    return {"id": str(tid), "cloud_id": str(cloud_id), "free_frac": fs["free_frac"],
            "occupied_cells": fs["occupied_cells"], "drivable_frac": dr["drivable_frac"],
            "surface": surf["surface"], "elevation_feature": elev["feature"], "max_slope": elev["max_slope"]}
