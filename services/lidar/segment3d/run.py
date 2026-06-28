"""Segment a stored cloud and persist point_segmentation. The native PTv3 path runs on the burst node; on
the interactive box it falls back to the projected segmentation (cuboids plus ground), which is the runnable,
ontology-consistent labeling for the camera fleet. Per-point labels (semantic class, instance id, confidence)
are written to the object store; the low-confidence fraction is recorded so uncertain regions are surfaced
for review rather than trusted blindly.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Object3D, PointCloud, PointSegmentation
from db.session import get_sessionmaker
from services.lidar.clean.ground import segment_ground
from services.lidar.ingest.store import load_cloud
from services.lidar.segment3d.semantic import (
    SegmentationUnavailable,
    segment_projected,
    segment_ptv3,
)

log = get_logger("lidar_segment3d")


async def segment_cloud(cloud_id: uuid.UUID, method: str | None = None) -> dict:
    """Segment a cloud into per-point semantic and instance labels and store point_segmentation."""
    cfg = get_settings().lidar
    method = method or cfg.segmenter
    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return {"error": "cloud not found", "cloud_id": str(cloud_id)}
        cuboids = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id))).scalars().all()
        session_id, cloud_uri = pc.session_id, pc.cloud_uri
    cub_list = [{"center": o.center, "dims": o.dims, "yaw": o.yaw, "class_id": o.class_id} for o in cuboids]

    cloud = load_cloud(cloud_uri)
    _, plane, _ = segment_ground(cloud)

    result, method_used, model_version = None, "projected_2d", "projected-3d-0.1"
    if method == "ptv3":
        try:
            result = segment_ptv3(cloud)
            method_used, model_version = "ptv3", cfg.segmenter_ckpt
        except SegmentationUnavailable as exc:
            log.info("lidar.ptv3_unavailable_fallback_projected", reason=str(exc))
    if result is None:
        result = segment_projected(cloud, cub_list, plane)

    buf = io.BytesIO()
    np.savez_compressed(buf, semantic=result["semantic"], instance=result["instance"], conf=result["conf"])
    store = get_object_store()
    store.ensure_bucket()
    uri = store.put_bytes(f"{cfg.cloud_prefix}/{session_id}/seg/{cloud_id}.npz", buf.getvalue(),
                          "application/octet-stream")

    async with get_sessionmaker()() as db:
        row = PointSegmentation(cloud_id=cloud_id, labels_uri=uri, model_version=model_version,
                                kind="panoptic", method=method_used, n_points=cloud.n,
                                low_conf_frac=result["low_conf_frac"])
        db.add(row)
        await db.flush()
        seg_id = row.seg_id
        await db.commit()
    log.info("lidar.segment_cloud", cloud=str(cloud_id), method=method_used, points=cloud.n,
             low_conf=round(result["low_conf_frac"], 3), instances=result["n_instances"])
    return {"seg_id": str(seg_id), "cloud_id": str(cloud_id), "method": method_used,
            "kind": "panoptic", "labels_uri": uri, "n_points": cloud.n,
            "classes_present": result["classes_present"], "n_instances": result["n_instances"],
            "low_conf_frac": round(result["low_conf_frac"], 4)}


def load_segmentation(labels_uri: str) -> dict:
    """Read stored per-point labels back."""
    store = get_object_store()
    with np.load(io.BytesIO(store.get_bytes(labels_uri))) as z:
        return {"semantic": z["semantic"], "instance": z["instance"], "conf": z["conf"]}
