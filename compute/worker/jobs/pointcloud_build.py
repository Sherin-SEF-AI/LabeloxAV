"""pointcloud_build: the burst job that lifts a session's synchronized camera frames into pseudo-LiDAR
clouds. Interactive work and small volumes run on the local 5080; large frame volumes burst to the A100
through the same compute seam as training (services/training/cloud.py), staged on the network volume with
the pod torn down after. The local path is fully wired here; the cloud target is the documented seam, not a
fake executor, exactly as the training cloud target is.

One pseudo-LiDAR cloud is built per synchronized frame group (the camera frames that share a ts_ns), placed
in the ego frame, and stored as a point_cloud row linked to those frames. Raw frames are never mutated.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
from pydantic import BaseModel
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame
from db.session import get_sessionmaker
from services.lidar.ingest.pseudo import lift_frame_group
from services.lidar.ingest.store import store_cloud

log = get_logger("pointcloud_build")

CALIB_VERSION = "labelox-calib-0.1"

_QUEUED_NOTE = (
    "Queued for the cloud A100. Provision the pod with `make cloud-provision`, push the session frames to "
    "/workspace on the network volume, run the pseudo-LiDAR lift on the pod, pull the clouds back into "
    "MinIO and write point_cloud rows, then stop the pod. The local worker will not run a cloud job."
)


class PointCloudBuildSpec(BaseModel):
    session_id: uuid.UUID
    ts_ns: list[int] | None = None          # specific frame groups; None = every group in the session
    source: str = "pseudo"                   # pseudo (lift cameras) is the only build source today
    model_id: str | None = None              # override the pinned depth model
    stride: int | None = None                # pixel stride for back-projection density
    max_points: int | None = None            # decimation ceiling per cloud
    compute_target: str = "local"            # local (5080) | cloud (A100 seam)


async def _frame_groups(session_id: uuid.UUID,
                        ts_filter: list[int] | None) -> list[tuple[int, dict[str, str]]]:
    """Group the session's real-camera frames by ts_ns into synchronized multi-camera groups."""
    cfg = get_settings()
    cams = set(cfg.rig.camera_lens.keys())
    async with get_sessionmaker()() as db:
        q = select(Frame.ts_ns, Frame.cam_id, Frame.img_uri).where(Frame.session_id == session_id)
        if ts_filter:
            q = q.where(Frame.ts_ns.in_(ts_filter))
        rows = (await db.execute(q.order_by(Frame.ts_ns))).all()
    groups: dict[int, dict[str, str]] = {}
    for ts_ns, cam_id, img_uri in rows:
        if cam_id in cams and img_uri:
            groups.setdefault(int(ts_ns), {})[cam_id] = img_uri
    return sorted(groups.items())


def _decode(img_uri: str) -> np.ndarray | None:
    buf = np.frombuffer(get_object_store().get_bytes(img_uri), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


async def build_session_clouds(spec: PointCloudBuildSpec) -> dict:
    """Local executor: lift every synchronized frame group in the session into a stored pseudo-LiDAR cloud."""
    groups = await _frame_groups(spec.session_id, spec.ts_ns)
    if not groups:
        return {"session_id": str(spec.session_id), "clouds": 0, "reason": "no camera frame groups"}

    built = []
    for ts_ns, cam_uris in groups:
        images: dict[str, np.ndarray] = {}
        for cam_id, uri in cam_uris.items():
            img = _decode(uri)
            if img is not None:
                images[cam_id] = img
        if not images:
            continue
        cloud = lift_frame_group(images, ts_ns=ts_ns, calibration_version=CALIB_VERSION,
                                 model_id=spec.model_id, stride=spec.stride, max_points=spec.max_points)
        res = await store_cloud(cloud, spec.session_id, source="pseudo",
                                depth_model=cloud.depth_model, calibration_version=CALIB_VERSION)
        built.append({"ts_ns": ts_ns, "cameras": sorted(images.keys()), **res})
        log.info("pointcloud_build.group", ts_ns=ts_ns, cameras=len(images), points=cloud.n)

    return {"session_id": str(spec.session_id), "clouds": len(built), "groups": built}


async def _queue_for_cloud(spec: PointCloudBuildSpec) -> dict:
    """The A100 seam: a cloud-target build parks with the documented data-movement contract. Not a fake
    executor; the local worker never claims it. Mirrors services/training/cloud.mark_queued_for_cloud."""
    log.info("pointcloud_build.queued_for_cloud", session=str(spec.session_id))
    return {"session_id": str(spec.session_id), "clouds": 0, "stage": "queued-cloud", "note": _QUEUED_NOTE}


async def run_pointcloud_build(spec: PointCloudBuildSpec) -> dict:
    """Dispatch by compute target: local runs on the 5080, cloud parks on the A100 seam."""
    if spec.compute_target == "cloud":
        return await _queue_for_cloud(spec)
    return await build_session_clouds(spec)
