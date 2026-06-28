"""Persist a normalized cloud: write the compressed npz to the object store and a point_cloud row, linked to
its session and, by the PPS ts_ns, to the camera frames captured at the same instant. Raw is immutable, so
this writes a new row and never mutates a cloud or a frame; cleaned variants are point_cloud_derived rows.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import ObjectStore, get_object_store
from db.models import Frame, PointCloud, PointCloudDerived
from db.session import get_sessionmaker
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_store")


async def store_cloud(cloud: Cloud, session_id: uuid.UUID, source: str | None = None,
                      depth_model: str | None = None, calibration_version: str | None = None,
                      store: ObjectStore | None = None) -> dict:
    """Write the cloud and its point_cloud row. Returns the cloud id plus the camera frames at the same ts_ns."""
    cfg = get_settings()
    store = store or get_object_store()
    store.ensure_bucket()
    uri = store.put_bytes(f"{cfg.lidar.cloud_prefix}/{session_id}/{cloud.ts_ns}.npz",
                          cloud.to_npz_bytes(), "application/octet-stream")
    async with get_sessionmaker()() as db:
        row = PointCloud(session_id=session_id, ts_ns=cloud.ts_ns, source=source or cloud.source,
                         cloud_uri=uri, point_count=cloud.n, depth_model=depth_model or cloud.depth_model,
                         calibration_version=calibration_version or cloud.calibration_version,
                         bounds=cloud.bounds())
        db.add(row)
        await db.flush()
        cid = row.cloud_id
        synced = (await db.execute(
            select(Frame.frame_id, Frame.cam_id)
            .where(Frame.session_id == session_id, Frame.ts_ns == cloud.ts_ns))).all()
        await db.commit()
    out = {"cloud_id": str(cid), "cloud_uri": uri, "point_count": cloud.n,
           "synced_frames": [{"frame_id": str(f), "cam_id": c} for f, c in synced]}
    log.info("lidar.stored", cloud=str(cid), session=str(session_id), points=cloud.n, synced=len(synced))
    return out


def load_cloud(cloud_uri: str, store: ObjectStore | None = None) -> Cloud:
    """Read a stored cloud back into the internal representation."""
    store = store or get_object_store()
    return Cloud.from_npz_bytes(store.get_bytes(cloud_uri))


async def store_derived(cloud_id: uuid.UUID, session_id: uuid.UUID, derived: Cloud, kind: str,
                        method: str, params: dict, store: ObjectStore | None = None) -> dict:
    """A cleaned or ground-removed variant. Raw is never overwritten: this is a new point_cloud_derived row."""
    cfg = get_settings()
    store = store or get_object_store()
    store.ensure_bucket()
    uri = store.put_bytes(f"{cfg.lidar.cloud_prefix}/{session_id}/derived/{cloud_id}_{kind}.npz",
                          derived.to_npz_bytes(), "application/octet-stream")
    async with get_sessionmaker()() as db:
        row = PointCloudDerived(cloud_id=cloud_id, kind=kind, uri=uri, method=method, params=params)
        db.add(row)
        await db.flush()
        did = row.derived_id
        await db.commit()
    log.info("lidar.derived", cloud=str(cloud_id), kind=kind, points=derived.n)
    return {"derived_id": str(did), "uri": uri, "kind": kind, "points": derived.n}
