"""Unify the 2D object, the 3D cuboid, and the multi-camera appearances into one identity. A lifted cuboid
already carries its 2D object (object_id); a native cuboid is matched to a 2D object by projecting it into the
synchronized camera and taking the best bbox IoU. object_3d.object_id is the unifying key: one physical
object across its 2D box, 2D mask, 3D cuboid, and every camera view.

The cross-camera link is the Phase 1 projection itself: a 3D box projects into each rig camera, so selecting
an object in the cloud highlights the same object in every synchronized view, and the reverse.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select, update

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, Object3D, PointCloud
from db.session import get_sessionmaker
from services.lidar.boxes import project_cuboid

log = get_logger("lidar_identity")


def projected_bbox(center, dims, yaw: float, cam_id: str, w: int, h: int,
                   pitch: float = 0.0, roll: float = 0.0) -> list[float] | None:
    """The 2D axis-aligned box of a cuboid's projection into a camera, clamped to the frame. None if the
    cuboid is not visible in front of that camera."""
    proj = project_cuboid(center, dims, yaw, cam_id, w, h, pitch, roll)
    uv = np.array(proj["corners_uv"], dtype=np.float32)
    vis = np.array(proj["in_front"], dtype=bool)
    if not vis.any():
        return None
    pts = uv[vis]
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-6 else 0.0


async def link_cloud(cloud_id: uuid.UUID, iou_thresh: float = 0.3) -> dict:
    """Link unlinked 3D cuboids on a cloud to the 2D objects across EVERY synchronized camera, by projecting
    each cuboid into each camera and taking the best bbox IoU. A fused cloud is shared by all cameras at its
    ts_ns, so a rear object links to a rear-camera 2D object, not just the front."""
    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return {"error": "cloud not found"}
        frames = (await db.execute(select(Frame).where(Frame.session_id == pc.session_id,
                  Frame.ts_ns == pc.ts_ns).order_by(Frame.cam_id))).scalars().all()
        if not frames:
            return {"cloud_id": str(cloud_id), "linked": 0, "reason": "no synchronized frames"}
        per_cam = []
        for fr in frames:
            objs = (await db.execute(select(Object).where(Object.frame_id == fr.frame_id))).scalars().all()
            per_cam.append((fr.cam_id, fr.width or 1280, fr.height or 960, objs))
        cuboids = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id,
                   Object3D.object_id.is_(None)))).scalars().all()

        linked = 0
        used: set[uuid.UUID] = set()
        for o3 in cuboids:
            best, best_iou = None, iou_thresh
            for cam, w, h, objs in per_cam:
                pbox = projected_bbox(o3.center, o3.dims, o3.yaw, cam, w, h, o3.pitch, o3.roll)
                if pbox is None:
                    continue
                for o2 in objs:
                    if o2.object_id in used or o2.class_id != o3.class_id:
                        continue
                    i = _iou(pbox, list(o2.bbox))
                    if i >= best_iou:
                        best, best_iou = o2, i
            if best is not None:
                o3.object_id = best.object_id
                used.add(best.object_id)
                linked += 1
        await db.commit()
    log.info("lidar.link_cloud", cloud=str(cloud_id), linked=linked, cameras=len(frames))
    return {"cloud_id": str(cloud_id), "linked": linked, "cuboids": len(cuboids)}


async def linked_views(object_3d_id: uuid.UUID) -> dict:
    """For a 3D object, the linked 2D object and the projection of the cuboid into every rig camera, so a
    selection in the cloud highlights the same physical object in all synchronized views."""
    cfg = get_settings()
    async with get_sessionmaker()() as db:
        o3 = await db.get(Object3D, object_3d_id)
        if o3 is None:
            return {"error": "object_3d not found"}
        pc = await db.get(PointCloud, o3.cloud_id)
        frame = (await db.execute(select(Frame).where(Frame.session_id == pc.session_id,
                 Frame.ts_ns == pc.ts_ns).order_by(Frame.cam_id).limit(1))).scalar_one_or_none()
        obj2d = await db.get(Object, o3.object_id) if o3.object_id else None
    w = frame.width if frame else 1280
    h = frame.height if frame else 960
    projections = {}
    for cam_id in cfg.rig.camera_lens:
        bbox = projected_bbox(o3.center, o3.dims, o3.yaw, cam_id, w, h, o3.pitch, o3.roll)
        if bbox is not None:
            projections[cam_id] = bbox
    return {"object_3d_id": str(object_3d_id), "object_id": str(o3.object_id) if o3.object_id else None,
            "class_id": o3.class_id, "projections": projections,
            "object_2d": ({"object_id": str(obj2d.object_id), "bbox": list(obj2d.bbox),
                           "cam_id": frame.cam_id if frame else None} if obj2d else None)}


async def linked_from_2d(object_id: uuid.UUID) -> dict:
    """The reverse: for a 2D object, the 3D cuboid it belongs to (if any) and its per-camera projections."""
    async with get_sessionmaker()() as db:
        o3 = (await db.execute(select(Object3D).where(Object3D.object_id == object_id).limit(1))
              ).scalar_one_or_none()
    if o3 is None:
        return {"object_id": str(object_id), "object_3d_id": None}
    return await linked_views(o3.object_3d_id)


async def assign_object_id(object_3d_id: uuid.UUID, object_id: uuid.UUID | None) -> dict:
    """Manually set or clear the 2D link on a 3D object (a human correction of the identity match)."""
    async with get_sessionmaker()() as db:
        await db.execute(update(Object3D).where(Object3D.object_3d_id == object_3d_id)
                         .values(object_id=object_id))
        await db.commit()
    return {"object_3d_id": str(object_3d_id), "object_id": str(object_id) if object_id else None}
