"""The 3D quality checker, the 3D equivalent of the 2D quality reviewer. 3D detection on pseudo-LiDAR
produces floating boxes, below-ground boxes, and impossible dimensions; this layer catches them before they
pollute the dataset. Detects: floating (box bottom above the ground), below-ground (bottom below it),
impossible dimensions (a 50 m car), duplicate (overlapping cuboids on the same points), misaligned (box not
enclosing its points), and missing neighbour (dense points with no box). Flags route to the same review and
active-learning loop as 2D.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from db.models import Object3D, QualityFlag3D
from db.session import get_sessionmaker
from services.lidar.boxes import ground_z, iou_3d
from services.lidar.clean.ground import segment_ground
from services.lidar.extract.common import cluster_dbscan, height_above_plane
from services.lidar.ingest.store import load_cloud
from services.lidar.segment3d.semantic import points_in_cuboid

log = get_logger("lidar_quality3d")


def check_cuboid(cuboid: dict, cloud_xyz: np.ndarray, plane: list[float],
                 neighbors: list[dict] | None = None) -> list[dict]:
    """Return the quality flags for one cuboid. cuboid is {center, dims, yaw}."""
    cfg = get_settings().lidar
    cx, cy, cz = cuboid["center"]
    length, width, height = cuboid["dims"]
    flags: list[dict] = []

    if max(length, width, height) > cfg.quality_max_dim_m or min(length, width, height) <= 0.0:
        flags.append({"kind": "impossible_dims", "score": round(max(length, width, height) / cfg.quality_max_dim_m, 2),
                      "detail": {"dims": [length, width, height]}})

    gz = ground_z(plane, cx, cy)
    bottom = cz - height / 2.0
    gap = bottom - gz
    if gap > cfg.quality_float_gap_m:
        flags.append({"kind": "floating", "score": round(min(1.0, gap / 2.0), 2), "detail": {"gap_m": round(gap, 2)}})
    elif gap < -cfg.quality_below_ground_m:
        flags.append({"kind": "below_ground", "score": round(min(1.0, -gap / 2.0), 2),
                      "detail": {"below_m": round(-gap, 2)}})

    n_in = int(points_in_cuboid(cloud_xyz, cuboid).sum())
    if n_in < 8:
        flags.append({"kind": "misaligned", "score": round(1.0 - n_in / 8.0, 2), "detail": {"points_in_box": n_in}})

    for nb in neighbors or []:
        if iou_3d(cuboid, nb) > cfg.quality_duplicate_iou:
            flags.append({"kind": "duplicate", "score": round(iou_3d(cuboid, nb), 2),
                          "detail": {"other_object_3d_id": nb.get("object_3d_id")}})
            break
    return flags


def _iou_2d(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 1e-6 else 0.0


def _projected_aabb(cuboid: dict, cam_id: str, w: int, h: int, calib=None) -> list[float] | None:
    """The axis-aligned 2D box of a cuboid's in-front corners projected into a camera, or None when the
    cuboid does not land in that image. calib is a resolved Calibration; None uses the nominal rig default."""
    from services.lidar.boxes import project_cuboid
    proj = project_cuboid(cuboid["center"], cuboid["dims"], cuboid["yaw"], cam_id, w, h,
                          cuboid.get("pitch", 0.0), cuboid.get("roll", 0.0), calib)
    uv = [c for c, infront in zip(proj["corners_uv"], proj["in_front"], strict=False) if infront]
    if len(uv) < 2 or not proj["any_in_image"]:
        return None
    xs, ys = [p[0] for p in uv], [p[1] for p in uv]
    return [min(xs), min(ys), max(xs), max(ys)]


def check_2d3d_consistency(cuboid: dict, views: list[dict], min_iou: float) -> dict | None:
    """Cross-sensor consistency: a 3D cuboid must reproject onto the 2D box of the same object in the cameras
    that see it. views is [{cam_id, w, h, bbox_2d}] for each camera with a 2D detection of this object. We
    project the cuboid into each and take the best 2D IoU; below min_iou in every camera the 3D and 2D boxes
    disagree (a bad lift or a wrong link), and we flag it. Returns a flag dict or None when consistent.

    Deliberately conservative (flag only when NO camera agrees): the fleet runs on nominal calibration, so a
    strict per-camera test would fire on projection noise. Tighten once real extrinsics land (the M-CALIB gap).
    """
    per_cam = []
    for v in views:
        if v.get("bbox_2d") is None:
            continue
        calib = v.get("calib")
        pb = _projected_aabb(cuboid, v["cam_id"], int(v["w"]), int(v["h"]), calib)
        if pb is None:
            continue
        per_cam.append({"cam_id": v["cam_id"], "iou": round(_iou_2d(pb, v["bbox_2d"]), 3),
                        "calib_source": getattr(calib, "source", "nominal")})
    if not per_cam:
        return None                                  # never jointly visible with a 2D box: nothing to check
    best = max(c["iou"] for c in per_cam)
    if best >= min_iou:
        return None                                  # at least one camera agrees -> consistent
    # the calibration source travels with the flag: a mismatch on nominal calibration is far weaker evidence
    # of a real labeling error than one on measured calibration.
    sources = sorted({c["calib_source"] for c in per_cam})
    return {"kind": "box_2d3d_inconsistent", "score": round(1.0 - best, 2),
            "detail": {"best_iou": best, "min_iou": min_iou, "per_cam": per_cam, "calib_sources": sources}}


async def check_object_consistency(object_3d_id: uuid.UUID, write: bool = True) -> dict:
    """Assemble the 2D views of a cuboid's linked object (its own camera plus any cross-camera views) and run
    the 2D-3D consistency check, optionally writing a QualityFlag3D. Returns the per-camera report."""
    from db.models import Frame, Object
    cfg = get_settings().lidar
    async with get_sessionmaker()() as db:
        o3d = await db.get(Object3D, object_3d_id)
        if o3d is None:
            return {"error": "object_3d not found"}
        cuboid = {"center": o3d.center, "dims": o3d.dims, "yaw": o3d.yaw, "pitch": o3d.pitch, "roll": o3d.roll}
        if o3d.object_id is None:
            return {"object_3d_id": str(object_3d_id), "checked": False, "reason": "no linked 2D object"}

        # the linked 2D object plus the same object in other rig cameras (cross_cam_links.views)
        object_ids = [o3d.object_id]
        link = (await db.get(Object, o3d.object_id))
        if link is not None and link.cross_cam_links:
            for oid in (link.cross_cam_links.get("views") or {}).values():
                if oid:
                    object_ids.append(uuid.UUID(str(oid)))
        views = []
        for oid in dict.fromkeys(object_ids):        # dedupe, keep order
            obj = await db.get(Object, oid)
            if obj is None:
                continue
            fr = await db.get(Frame, obj.frame_id)
            if fr is None:
                continue
            views.append({"cam_id": fr.cam_id, "w": fr.width, "h": fr.height, "bbox_2d": list(obj.bbox),
                          "session_id": fr.session_id})

    # resolve each camera's calibration (stored real, else nominal) so the projection uses real calibration
    # when the session has it; the source is recorded with the verdict.
    from services.calibration.resolve import resolve_calibration
    for v in views:
        v["calib"] = await resolve_calibration(v["session_id"], v["cam_id"], int(v["w"]), int(v["h"]))

    flag = check_2d3d_consistency(cuboid, views, cfg.quality_2d3d_min_iou)
    if flag and write:
        async with get_sessionmaker()() as db:
            db.add(QualityFlag3D(object_3d_id=object_3d_id, cloud_id=o3d.cloud_id, kind=flag["kind"],
                                 score=flag["score"], detail=flag["detail"]))
            await db.commit()
    log.info("lidar.consistency", object_3d=str(object_3d_id), cameras=len(views),
             flagged=bool(flag))
    return {"object_3d_id": str(object_3d_id), "checked": True, "cameras": len(views),
            "consistent": flag is None, "flag": flag}


def find_missing_neighbors(cloud_xyz: np.ndarray, plane: list[float], cuboids: list[dict],
                           min_cluster: int = 60) -> list[dict]:
    """Dense non-ground clusters with no cuboid over them: a likely missed object."""
    pts = cloud_xyz[height_above_plane(cloud_xyz, plane) > 0.3]
    if len(pts) < min_cluster:
        return []
    labels = cluster_dbscan(pts, eps=0.8)
    missing = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = pts[labels == cl]
        if len(cxyz) < min_cluster:
            continue
        centroid = cxyz.mean(axis=0)
        covered = any(points_in_cuboid(centroid[None, :], c)[0] for c in cuboids)
        if not covered:
            missing.append({"kind": "missing_neighbor", "score": round(min(1.0, len(cxyz) / 300.0), 2),
                            "detail": {"centroid": [round(float(x), 2) for x in centroid], "n_points": int(len(cxyz))}})
    return missing


async def check_cloud(cloud_id: uuid.UUID) -> dict:
    """Run the 3D quality checker over a cloud's objects and write quality_flag_3d rows."""
    async with get_sessionmaker()() as db:
        from db.models import PointCloud
        pc = await db.get(PointCloud, cloud_id)
        if pc is None:
            return {"error": "cloud not found"}
        objs = (await db.execute(select(Object3D).where(Object3D.cloud_id == cloud_id))).scalars().all()
        cloud_uri = pc.cloud_uri
    cloud = load_cloud(cloud_uri)
    _, plane, _ = segment_ground(cloud)
    cubs = [{"object_3d_id": str(o.object_3d_id), "center": o.center, "dims": o.dims, "yaw": o.yaw}
            for o in objs]

    written = 0
    async with get_sessionmaker()() as db:
        for i, o in enumerate(objs):
            neighbors = [c for j, c in enumerate(cubs) if j != i]
            for f in check_cuboid(cubs[i], cloud.xyz, plane, neighbors):
                db.add(QualityFlag3D(object_3d_id=o.object_3d_id, cloud_id=cloud_id, kind=f["kind"],
                                     score=f["score"], detail=f["detail"]))
                written += 1
        for f in find_missing_neighbors(cloud.xyz, plane, cubs):
            db.add(QualityFlag3D(object_3d_id=None, cloud_id=cloud_id, kind=f["kind"], score=f["score"],
                                 detail=f["detail"]))
            written += 1
        await db.commit()

    # cross-sensor 2D-3D consistency reads the 2D views (frames and linked objects), not the cloud, so it
    # runs as its own pass over the linked objects rather than inside the geometry loop.
    consistency = 0
    for o in objs:
        if o.object_id is not None:
            res = await check_object_consistency(o.object_3d_id, write=True)
            if res.get("flag"):
                consistency += 1
    written += consistency

    log.info("lidar.quality3d", cloud=str(cloud_id), objects=len(objs), flags=written, consistency=consistency)
    return {"cloud_id": str(cloud_id), "objects": len(objs), "flags": written, "consistency_flags": consistency}


async def confirm_flag(flag_id: uuid.UUID) -> dict:
    """Confirm a 3D quality flag: demote the flagged object back to review (the same loop as 2D)."""
    async with get_sessionmaker()() as db:
        flag = await db.get(QualityFlag3D, flag_id)
        if flag is None:
            return {"error": "flag not found"}
        flag.status = "confirmed"
        if flag.object_3d_id:
            o = await db.get(Object3D, flag.object_3d_id)
            if o is not None:
                o.state = "review"
                o.version += 1
        await db.commit()
    log.info("lidar.quality3d_confirm", flag=str(flag_id), kind=flag.kind)
    return {"flag_id": str(flag_id), "status": "confirmed", "kind": flag.kind}
