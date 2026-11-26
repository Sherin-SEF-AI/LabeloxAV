"""Point labels a person paints on /lidar/segment, and the one-click helpers that seed them.

A person's labels are a new `point_segmentation` row with source 'human' and a version that counts up per
cloud; the projected fallback (source 'auto') and a segmenter's output (source 'model') are never rewritten.
Each save is a full per-point label array, so any version can be read back on its own; `base_seg_id` records
which labels the edit started from.

The browser works on a deterministic subset of the cloud, every `stride`-th point, so a 1.2M-point
pseudo-LiDAR cloud stays interactive. Labels sent back for that subset are merged into the full array of the
base version: points the browser never saw keep the base's label rather than being blanked.

Helpers (each returns a boolean mask over the full cloud):
- ground: the ground plane (services/lidar/clean/ground.py), validated to be near-horizontal;
- cluster: the DBSCAN cluster (services/lidar/extract/common.py) of non-ground points under a clicked point;
- from 2D: every point that projects into a frame's semantic mask or an object's polygon takes that class;
- from cuboids: the projected segmentation (segment3d/semantic.py segment_projected) of the cloud's cuboids.
"""

from __future__ import annotations

import io
import json
import math
import uuid

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud
from services.lidar.segment3d.semantic import BACKGROUND

log = get_logger("lidar_segment_human")

# Points sent to the browser at most. KITTI scans are ~120k and go whole; pseudo-LiDAR clouds are thinned.
MAX_BROWSER_POINTS = 400_000


def stride_for(n: int, max_points: int = MAX_BROWSER_POINTS) -> int:
    """The subset step: every point when the cloud fits, else the smallest step that brings it under."""
    return max(1, math.ceil(n / max(1, max_points)))


def pack_labels(semantic: np.ndarray, instance: np.ndarray | None = None, conf: np.ndarray | None = None) -> bytes:
    n = len(semantic)
    buf = io.BytesIO()
    np.savez_compressed(buf, semantic=semantic.astype(np.int32),
                        instance=(instance if instance is not None else np.full(n, BACKGROUND)).astype(np.int32),
                        conf=(conf if conf is not None else np.where(semantic >= 0, 1.0, 0.0)).astype(np.float32),
                        object_3d_ids=np.array([], dtype="U36"))
    return buf.getvalue()


def merge_subset(base: np.ndarray, subset_labels: np.ndarray, stride: int) -> np.ndarray:
    """The base labels with the browser's subset written over them (subset point i is cloud point i*stride)."""
    out = base.astype(np.int32).copy()
    idx = np.arange(len(subset_labels), dtype=np.int64) * stride
    if len(idx) and idx[-1] >= len(out):
        raise ValueError(f"{len(subset_labels)} labels at stride {stride} do not fit a {len(out)}-point cloud")
    out[idx] = subset_labels.astype(np.int32)
    return out


async def latest_segmentation(db: AsyncSession, cloud_id: uuid.UUID, source: str | None = None):
    from db.models import PointSegmentation

    q = select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id)
    if source:
        q = q.where(PointSegmentation.source == source)
    return (await db.execute(q.order_by(PointSegmentation.created_at.desc(), PointSegmentation.version.desc())
                             .limit(1))).scalars().first()


def full_labels(row, n: int) -> np.ndarray:
    """A segmentation row's per-point class ids, or all-unlabelled when the row is missing or misaligned."""
    if row is None:
        return np.full(n, BACKGROUND, dtype=np.int32)
    from services.lidar.segment3d.run import load_segmentation

    sem = np.asarray(load_segmentation(row.labels_uri)["semantic"], dtype=np.int32)
    if len(sem) != n:
        log.warning("segment_human.misaligned", seg=str(row.seg_id), labels=len(sem), points=n)
        return np.full(n, BACKGROUND, dtype=np.int32)
    return sem


async def save_human_labels(db: AsyncSession, cloud_id: uuid.UUID, subset_labels: np.ndarray, stride: int, *,
                            base_seg_id: uuid.UUID | None = None, user_id: uuid.UUID | None = None) -> dict:
    """Write a new human version of a cloud's point labels. Commits."""
    from core.config import get_settings
    from core.storage import get_object_store
    from db.models import PointCloud, PointSegmentation
    from services.lidar.ingest.store import load_cloud

    pc = await db.get(PointCloud, cloud_id)
    if pc is None:
        raise LookupError("cloud not found")
    cloud = load_cloud(pc.cloud_uri)
    base_row = None
    if base_seg_id is not None:
        base_row = await db.get(PointSegmentation, base_seg_id)
        if base_row is None or base_row.cloud_id != cloud_id:
            raise LookupError("base segmentation not found on this cloud")
    base = full_labels(base_row, cloud.n)
    sem = merge_subset(base, np.asarray(subset_labels), stride)
    version = int((await db.execute(select(func.coalesce(func.max(PointSegmentation.version), 0)).where(
        PointSegmentation.cloud_id == cloud_id, PointSegmentation.source == "human"))).scalar() or 0) + 1
    store = get_object_store()
    store.ensure_bucket()
    uri = store.put_bytes(f"{get_settings().lidar.cloud_prefix}/{pc.session_id}/seg/{cloud_id}.human.v{version}.npz",
                          pack_labels(sem), "application/octet-stream")
    labelled = int((sem >= 0).sum())
    row = PointSegmentation(cloud_id=cloud_id, labels_uri=uri, model_version="human", kind="semantic",
                            method="human", n_points=cloud.n,
                            low_conf_frac=float(1.0 - labelled / cloud.n) if cloud.n else 0.0,
                            source="human", version=version, base_seg_id=base_seg_id, created_by=user_id)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    classes = {int(c): int(k) for c, k in zip(*np.unique(sem[sem >= 0], return_counts=True), strict=True)}
    log.info("segment_human.saved", cloud=str(cloud_id), version=version, labelled=labelled, points=cloud.n)
    return {"seg_id": str(row.seg_id), "version": version, "labelled": labelled, "n_points": cloud.n,
            "classes": classes}


def ground_mask(cloud: Cloud) -> tuple[np.ndarray, list[float], str]:
    from services.lidar.clean.ground import segment_ground

    return segment_ground(cloud)


def cluster_at(cloud: Cloud, point, *, eps: float = 0.5, min_points: int = 8, radius_m: float = 12.0,
               pick_m: float = 1.0, plane: list[float] | None = None) -> tuple[np.ndarray, dict]:
    """The DBSCAN cluster of non-ground points under a clicked 3D point, as a mask over the full cloud.

    Clustering runs on the non-ground points within `radius_m` of the click only: DBSCAN over a whole
    120k-point scan for one click is seconds, over the neighbourhood it is milliseconds, and an object larger
    than 24 m across is not one a click is selecting.
    """
    from services.lidar.extract.common import cluster_dbscan, nonground_mask

    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    if plane is None:
        _, plane, _ = ground_mask(cloud)
    near = np.linalg.norm(xyz - p, axis=1) <= radius_m
    cand = near & nonground_mask(cloud, plane, thresh=0.15)
    idx = np.flatnonzero(cand)
    mask = np.zeros(cloud.n, dtype=bool)
    if len(idx) == 0:
        return mask, {"reason": "no non-ground points near the click"}
    d = np.linalg.norm(xyz[idx] - p, axis=1)
    seed = int(np.argmin(d))
    if d[seed] > pick_m:
        return mask, {"reason": f"no non-ground point within {pick_m:g} m of the click"}
    labels = cluster_dbscan(xyz[idx], eps=eps, min_points=min_points)
    lab = int(labels[seed])
    if lab < 0:
        mask[idx[seed]] = True
        return mask, {"reason": "the clicked point is DBSCAN noise; selected it alone", "points": 1}
    mask[idx[labels == lab]] = True
    sel = xyz[mask]
    return mask, {"points": int(mask.sum()), "eps": eps,
                  "extent_m": [round(float(v), 2) for v in (sel.max(axis=0) - sel.min(axis=0))]}


def _polygon_mask(blob: dict, height: int, width: int) -> np.ndarray | None:
    import cv2

    if blob.get("encoding") != "polygon":
        return None
    mw, mh = int(blob.get("width") or width), int(blob.get("height") or height)
    m = np.zeros((mh, mw), dtype=np.uint8)
    for poly in blob.get("polygons") or []:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if (mh, mw) != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


async def labels_from_2d(db: AsyncSession, cloud_id: uuid.UUID, cloud: Cloud,
                         max_depth_m: float = 80.0) -> tuple[np.ndarray, dict]:
    """Per-point classes copied from the 2D masks of the camera frames captured with this scan.

    Two sources, the more specific winning: the frame's semantic segmentation raster (road, sidewalk,
    vegetation...), then each object's polygon mask (a car's points take the car's class). A point takes a
    class only if it projects in front of the camera and inside the image through the session's resolved
    calibration; the rest stay unlabelled (-1). Occlusion is not modelled, which is why the result is a
    suggestion the person reviews, not a save.
    """
    from core.storage import get_object_store
    from db.models import Frame, FrameSegmentation, Object, PointCloud
    from services.calibration.resolve import resolve_calibration
    from services.lidar.project import project_to_camera

    pc = await db.get(PointCloud, cloud_id)
    frames = (await db.execute(select(Frame).where(Frame.session_id == pc.session_id,
                                                   Frame.ts_ns == pc.ts_ns))).scalars().all()
    out = np.full(cloud.n, BACKGROUND, dtype=np.int32)
    report: dict = {"frames": len(frames), "semantic_frames": 0, "objects": 0}
    if not frames:
        report["reason"] = "no camera frame was captured with this scan"
        return out, report
    store = get_object_store()
    xyz = np.asarray(cloud.xyz, dtype=np.float32)
    for f in frames:
        cal = await resolve_calibration(pc.session_id, f.cam_id, int(f.width), int(f.height))
        proj = project_to_camera(xyz, f.cam_id, int(f.width), int(f.height), cal)
        ok = proj["in_image"] & (proj["depth"] < max_depth_m)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        u = np.clip(proj["uv"][idx, 0].astype(np.int64), 0, int(f.width) - 1)
        v = np.clip(proj["uv"][idx, 1].astype(np.int64), 0, int(f.height) - 1)
        seg = (await db.execute(select(FrameSegmentation).where(
            FrameSegmentation.frame_id == f.frame_id, FrameSegmentation.kind == "semantic")
            .order_by((FrameSegmentation.source == "human").desc(), FrameSegmentation.created_at.desc())
            .limit(1))).scalars().first()
        if seg is not None:
            try:
                with np.load(io.BytesIO(store.get_bytes(seg.labels_uri))) as z:
                    raster = z["arr"] if "arr" in z.files else z[z.files[0]]
                if raster.shape[:2] != (int(f.height), int(f.width)):
                    import cv2
                    raster = cv2.resize(raster.astype(np.int32), (int(f.width), int(f.height)),
                                        interpolation=cv2.INTER_NEAREST)
                cls = raster[v, u].astype(np.int32)
                good = cls > 0
                out[idx[good]] = cls[good]
                report["semantic_frames"] += 1
            except Exception as exc:  # noqa: BLE001 - one unreadable raster skips that layer, not the copy
                log.info("segment_human.raster_unreadable", frame=str(f.frame_id), error=str(exc)[:120])
        objs = (await db.execute(select(Object).where(
            Object.frame_id == f.frame_id, Object.mask_uri.is_not(None), Object.state != "rejected"))).scalars().all()
        # larger masks first so a smaller object in front (a rider on a bike) wins the overlap
        def _area(o) -> float:
            b = o.bbox or [0, 0, 0, 0]
            return (b[2] - b[0]) * (b[3] - b[1])
        for o in sorted(objs, key=_area, reverse=True):
            try:
                blob = json.loads(store.get_bytes(o.mask_uri).decode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            m = _polygon_mask(blob, int(f.height), int(f.width))
            if m is None:
                continue
            hit = m[v, u]
            out[idx[hit]] = int(o.class_id)
            report["objects"] += 1
    report["labelled"] = int((out >= 0).sum())
    return out, report


async def labels_from_cuboids(db: AsyncSession, cloud_id: uuid.UUID, cloud: Cloud) -> tuple[np.ndarray, dict]:
    """The projected segmentation: points in a (non-rejected) cuboid take its class, ground points are road."""
    from db.models import Object3D
    from services.lidar.segment3d.semantic import segment_projected

    cuboids = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id,
                                                       Object3D.state != "rejected"))).scalars().all()
    _, plane, _ = ground_mask(cloud)
    res = segment_projected(cloud, [{"center": o.center, "dims": o.dims, "yaw": o.yaw, "class_id": o.class_id}
                                    for o in cuboids], plane)
    return res["semantic"].astype(np.int32), {"cuboids": len(cuboids), "labelled": int((res["semantic"] >= 0).sum())}
