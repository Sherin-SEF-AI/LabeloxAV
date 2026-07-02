"""Annotate-once, propagate (M-MC.3, Tier 2): draw an object once in one camera and place it in the other rig
views by geometry. This tier is gated on per-session calibration (CalibrationValidation): projecting a box
across cameras needs real intrinsics and extrinsics, so a session that has not passed M-CAL validation falls
back to Tier 1 (manual linking) and this returns gated=True.

The method is lens-aware and reuses the Phase 3 projection stack (services/lidar/project.py):

  1. lift the source object's ground-contact pixel (bottom-centre of the box) to a 3D point on the road plane
     in the ego frame, through the SOURCE camera's resolved calibration (fisheye undistort for the wide lenses);
  2. project that 3D point, and the point one estimated object-height above it, into each TARGET camera through
     its own resolved calibration (pinhole or fisheye) to get a seed box;
  3. optionally snap the seed with a SAM box prompt on the target frame;
  4. also compute the epipolar line of the source pixel in the target view (project the source ray at a near and
     a far depth) so the search is constrained and the UI can show the geometry.

Every propagated object is stamped source="propagated" with provenance pointing at the origin object, routed to
review (a human confirms the geometry), and linked into the source's rig identity (M-MC.2).
"""

from __future__ import annotations

import math
from uuid import UUID

import numpy as np
from sqlalchemy import select

from core.logging import get_logger
from db.models import Frame, FrameGroup, Object
from db.session import get_sessionmaker
from services.calibration.report import session_calibrated
from services.calibration.resolve import resolve_calibration
from services.lidar.project import project_to_camera

log = get_logger("multicam.propagate")


def _lift_ground(u: float, v: float, calib) -> np.ndarray | None:
    """A pixel in this camera to a 3D point on the road plane (ego z=0), through the resolved calibration.
    Fisheye lenses undistort first, so the ray is correct off-centre. None if the ray never meets the ground."""
    import cv2

    if calib.model == "fisheye" and calib.dist:
        km = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]], dtype=np.float64)
        d = np.array((list(calib.dist) + [0, 0, 0, 0])[:4], dtype=np.float64).reshape(4, 1)
        und = cv2.fisheye.undistortPoints(np.array([[[float(u), float(v)]]], dtype=np.float64), km, d)
        dir_cam = np.array([und[0, 0, 0], und[0, 0, 1], 1.0], dtype=np.float32)
    else:
        dir_cam = np.array([(u - calib.cx) / calib.fx, (v - calib.cy) / calib.fy, 1.0], dtype=np.float32)
    origin = calib.t().astype(np.float32)                 # camera centre in the ego frame
    dir_ego = (calib.R() @ dir_cam).astype(np.float32)    # cam = (ego - t) @ R, R orthonormal -> ego_dir = R @ cam_dir
    if abs(dir_ego[2]) < 1e-6 or (dir_ego[2] >= 0 and origin[2] > 0):
        return None                                       # ray parallel to or rising away from the ground
    s = -origin[2] / dir_ego[2]
    if s <= 0:
        return None
    return origin + s * dir_ego


def _project(point: np.ndarray, cam_id: str, w: int, h: int, calib) -> tuple[float, float] | None:
    r = project_to_camera(point.reshape(1, 3), cam_id, w, h, calib)
    if not bool(r["in_front"][0]):
        return None
    return float(r["uv"][0, 0]), float(r["uv"][0, 1])


async def propagate_object(object_id: UUID, use_sam: bool = True) -> dict:
    """Place a source object into the other rig views by projection. Gated on calibration; returns gated=True
    (tier 1) if the session has not passed validation."""
    maker = get_sessionmaker()
    async with maker() as db:
        src = await db.get(Object, object_id)
        if src is None:
            return {"error": "object not found"}
        frame = await db.get(Frame, src.frame_id)
        session_id = frame.session_id
        if not await session_calibrated(session_id):
            return {"gated": True, "tier": 1,
                    "reason": "session not calibrated: use Tier 1 manual linking (run calibration validation to enable projection)"}

        groups = (await db.execute(
            select(FrameGroup).where(FrameGroup.session_id == session_id).order_by(FrameGroup.ts_ns))).scalars().all()
        # pick the group whose frame_ids include this frame
        grp = next((g for g in groups if str(src.frame_id) in (g.frame_ids or {}).values()), None)
        if grp is None:
            return {"error": "no frame group for this frame (build groups first)"}
        src_cam = frame.cam_id
        targets = {cam: fid for cam, fid in (grp.frame_ids or {}).items() if cam != src_cam}
        if not targets:
            return {"error": "no other cameras in this group to propagate to"}

        src_calib = await resolve_calibration(session_id, src_cam, frame.width, frame.height)
        x0, y0, x1, y1 = [float(c) for c in src.bbox]
        base_px = ((x0 + x1) / 2.0, y1)
        p_base = _lift_ground(*base_px, src_calib)
        if p_base is None:
            return {"error": "the object's ground contact does not meet the road plane (projection needs a ground object)"}
        dist_src = float(np.linalg.norm(p_base - src_calib.t()))
        real_h = (y1 - y0) / src_calib.fy * dist_src      # metric height from the box height and range
        real_w = (x1 - x0) / src_calib.fx * dist_src
        p_top = p_base + np.array([0.0, 0.0, real_h], dtype=np.float32)

        # resolve every target frame's meta once
        frame_rows = {str(f.frame_id): f for f in (await db.execute(
            select(Frame).where(Frame.frame_id.in_([UUID(fid) for fid in targets.values()])))).scalars().all()}

        results = []
        created = []
        for cam, fid in targets.items():
            tf = frame_rows.get(fid)
            if tf is None:
                continue
            tcal = await resolve_calibration(session_id, cam, tf.width, tf.height)
            base_uv = _project(p_base, cam, tf.width, tf.height, tcal)
            if base_uv is None:
                results.append({"cam": cam, "in_view": False, "reason": "projects behind or outside the target camera"})
                continue
            top_uv = _project(p_top, cam, tf.width, tf.height, tcal) or (base_uv[0], base_uv[1] - 1)
            dist_tgt = float(np.linalg.norm(p_base - tcal.t())) or 1.0
            box_h = abs(base_uv[1] - top_uv[1]) or (real_h * tcal.fy / dist_tgt)
            box_w = real_w * tcal.fx / dist_tgt
            uc = base_uv[0]
            box = [uc - box_w / 2.0, min(base_uv[1], top_uv[1]), uc + box_w / 2.0, max(base_uv[1], top_uv[1])]
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(tf.width), box[2]), min(float(tf.height), box[3])]

            epiline = _epipolar_line(base_px, src_cam, frame, cam, tf, src_calib, tcal, session_id)

            sam_used = False
            if use_sam:
                snapped = _sam_snap(tf.img_uri, box)
                if snapped is not None:
                    box, sam_used = snapped[0], True

            results.append({"cam": cam, "in_view": True, "box": [round(c, 1) for c in box],
                            "projected_base": [round(base_uv[0], 1), round(base_uv[1], 1)],
                            "epipolar": epiline, "sam_used": sam_used})
            created.append({"frame_id": UUID(fid), "cam": cam, "box": box, "sam_used": sam_used})

        # write the propagated objects + rig-link them to the source
        new_ids = []
        for c in created:
            o = Object(frame_id=c["frame_id"], class_id=src.class_id, bbox=[float(x) for x in c["box"]],
                       conf=float(src.conf or 0.5), source="propagated", state="review",
                       provenance={"from_object_id": str(object_id), "method": "ground_project",
                                   "sam": c["sam_used"], "src_cam": src_cam})
            db.add(o)
            await db.flush()
            new_ids.append((o.object_id, c["cam"]))
        await db.commit()

    # link each propagated object into the source's rig identity (M-MC.2), reusing that tier's merge/vote
    if new_ids:
        from services.multicam.rigident import link_objects

        for oid, _cam in new_ids:
            await link_objects(session_id, grp.group_id, [object_id, oid], source="projection")

    out = {"source_object_id": str(object_id), "src_cam": src_cam, "metric": {"height_m": round(real_h, 2),
           "width_m": round(real_w, 2), "range_m": round(dist_src, 1)}, "targets": results,
           "created": [str(o) for o, _ in new_ids]}
    log.info("multicam.propagated", object_id=str(object_id), created=len(new_ids))
    return out


def _epipolar_line(src_px, src_cam, src_frame, tgt_cam, tgt_frame, src_calib, tgt_calib, session_id) -> list | None:
    """The source pixel's epipolar line in the target view: project the source ray at a near and a far depth.
    Constrains the SAM search and lets the UI draw the geometric constraint even when the ground lift is weak."""
    if src_calib.model == "fisheye" and src_calib.dist:
        return None  # ray direction handled in _lift_ground; skip the display line for the wide lens for now
    u, v = src_px
    dir_cam = np.array([(u - src_calib.cx) / src_calib.fx, (v - src_calib.cy) / src_calib.fy, 1.0], dtype=np.float32)
    origin = src_calib.t().astype(np.float32)
    dir_ego = (src_calib.R() @ dir_cam).astype(np.float32)
    dir_ego /= (np.linalg.norm(dir_ego) or 1.0)
    pts = []
    for depth in (3.0, 40.0):
        uv = _project(origin + depth * dir_ego, tgt_cam, tgt_frame.width, tgt_frame.height, tgt_calib)
        if uv is not None:
            pts.append([round(uv[0], 1), round(uv[1], 1)])
    return pts if len(pts) == 2 else None


def _sam_snap(img_uri: str, box: list[float]) -> list[float] | None:
    """Best-effort SAM box-prompt snap of a projected box on the target frame. Returns the snapped bbox or None
    (SAM unavailable, GPU busy, or empty mask) so propagation degrades cleanly to the projected box."""
    try:
        import cv2

        from core.storage import get_object_store
        from services.api.sam_service import segment as run_segment

        buf = np.frombuffer(get_object_store().get_bytes(img_uri), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        res = run_segment(img, box=[float(c) for c in box])
        if res.get("bbox"):
            return [float(c) for c in res["bbox"]]
    except Exception as exc:  # noqa: BLE001 - propagation must survive a missing GPU
        log.info("multicam.sam_snap_skipped", error=str(exc))
    return None
