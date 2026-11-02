"""Propagating a label to the next frame with the ego motion, and refusing when the pose is not there.

Ego-compensated propagation needs a rigid transform between two frames: how the camera moved. Until
2026-11-02 this corpus had GNSS on 3 frames of 41,752 and the guard below refused on every session. The
dashcam stamp reader (`services/ingest/overlay_telemetry.py`) now gives most drives a measured trajectory,
so the guard passes there, and the first measurement of what that buys is in docs/MEASUREMENT.md (row 5):
on these uncalibrated dashcams a warped box lands on the object less often than a plain copy does, so this
stays an explicit, dry-run-by-default tool rather than a step in any default path.

Two ways of moving a box, and they check each other:

    geometry   the ground homography the ego motion induces (core/accel/ego_homography.py). Exact for a
               static object on the road surface, meaningless for anything else, which is why the pack's
               motion model gates it.
    tracker    whatever the tracker already produced for that object on the destination frame.

When both exist and agree within tolerance, the box is written and stamped with the run that produced it.
When they disagree, NEITHER is written and a PropagationConflict row is: picking one would propagate a
guess, and averaging them would propagate a box neither method proposed. The resulting table names the
frames where the calibration or the pose is wrong, which is worth more than the label would have been.

Provenance goes in `Object.provenance` under `derived_from_run`, not in a new column. `provenance` already
carries run stamps (`agent_run_id` among them), and grep shows `derived_from_run` appears nowhere in the
tree despite being referenced in the brief, so there is no existing column to honour.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.boxes import box_iou_matrix
from core.accel.ego_homography import ground_homography, warp_box
from core.logging import get_logger
from db.models import Frame, Object, PropagationConflict

log = get_logger("ego_propagate")

# How far the two methods may disagree, as IoU between the boxes they propose. Below this the two are
# describing different objects and neither should be trusted.
AGREE_IOU = 0.5


class PoseUnavailable(Exception):
    """Raised rather than returned, because a caller that ignores this writes wrong labels silently."""


async def ego_transform(db: AsyncSession, from_frame: Frame, to_frame: Frame) -> dict[str, Any]:
    """The camera motion between two frames, or a refusal naming what is missing.

    Returns {"measured", "R", "t", "reason"}, in the vehicle frame of `from_frame`. Where a frame has no
    pose the module says which of the two lacked what rather than reporting a generic failure.
    """
    # The pose table (0112) is asked first: `services/intelligence/ego_pose` builds it from GNSS where a
    # drive has it and from the images where it does not. A pose pair gives both rotation and translation,
    # which is what propagation actually needs.
    poses = await _pose_pair(db, from_frame, to_frame)
    if poses is not None:
        a, b = poses
        return _transform_from_poses(a, b)

    missing = []
    for label, f in (("from", from_frame), ("to", to_frame)):
        if f.gnss is None:
            missing.append(f"{label} frame has no GNSS fix")
    if missing:
        return {"measured": False, "R": None, "t": None,
                "reason": "; ".join(missing) + ". Ego-compensated propagation needs the camera motion "
                          "between the two frames, and neither frame has a row in the ego_pose table "
                          "either, so the motion between them is unknown"}

    # With both fixes present but no pose rows the translation is derivable and the rotation is not, and a
    # translation-only transform is not offered.
    return {"measured": False, "R": None, "t": None,
            "reason": "both frames carry GNSS but no heading or IMU attitude, so the rotation between "
                      "them is unknown; a translation-only transform would place a rotated scene wrongly"}


async def _pose_pair(db: AsyncSession, a: Frame, b: Frame):
    """The two frames' ego poses, or None when either is missing one."""
    from db.models import EgoPose

    rows = (await db.execute(select(EgoPose).where(
        EgoPose.session_id == a.session_id,
        EgoPose.ts_ns.in_((int(a.ts_ns), int(b.ts_ns)))))).scalars().all()
    by_ts = {int(r.ts_ns): r for r in rows}
    pa, pb = by_ts.get(int(a.ts_ns)), by_ts.get(int(b.ts_ns))
    return (pa, pb) if (pa is not None and pb is not None) else None


def _transform_from_poses(a, b) -> dict[str, Any]:
    """The rigid transform from one pose to the next, and whether either end was actually measured.

    `measured` is the conjunction on purpose: a transform between a measured pose and an inferred one is
    an inferred transform, and calling it measured because half of it was would be the kind of quiet
    upgrade this whole table exists to prevent. Yaw only, because that is all either source observes.
    """
    import math

    ya = 2.0 * math.atan2(a.qz, a.qw)
    yb = 2.0 * math.atan2(b.qz, b.qw)
    dyaw = (yb - ya + math.pi) % (2 * math.pi) - math.pi
    c, s = math.cos(dyaw), math.sin(dyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    # World displacement rotated into the first frame's ego axes.
    dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
    t = np.array([dx * math.cos(-ya) - dy * math.sin(-ya),
                  dx * math.sin(-ya) + dy * math.cos(-ya), dz], dtype=float)
    measured = bool(a.measured and b.measured)
    quality = min(float(a.quality or 0.0), float(b.quality or 0.0))
    return {"measured": measured, "R": R, "t": t, "quality": quality,
            "pose_source": f"{a.source}->{b.source}",
            "reason": None if measured else
                      (f"both poses come from {a.source} and were inferred rather than observed "
                       f"(quality {quality:.2f}); the transform is usable but is not a measurement")}


def camera_motion(motion: dict[str, Any], calib) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The pose change as `ground_homography` wants it: in the source camera's optical axes.

    The pose transform is in the vehicle frame (x forward, y left, z up) and describes how the VEHICLE
    moved; the homography wants rotation of the scene seen by the camera and the plane in optical axes
    (x right, y down, z forward). Passing the vehicle-frame values straight through reads forward motion as
    sideways and turns the wrong way, which is what this module did before it had ever run on a real pose.

    Returns (R, t, normal, height) where `t` is already premultiplied by R, so that the primitive's
    `R - t n^T / h` equals the exact `R (I - t_c n^T / h)` for a camera that both turned and moved.
    """
    from services.calibration.resolve import ipm_args

    R_ego = np.asarray(motion["R"], dtype=np.float64)
    t_ego = np.asarray(motion["t"], dtype=np.float64).reshape(3)
    M = np.asarray(calib.R(), dtype=np.float64).T          # vehicle column vector -> optical
    mount = np.asarray(calib.xyz_m, dtype=np.float64)
    # The camera centre moves with the vehicle plus the swing of its mount about the vehicle origin.
    t_cam = M @ (t_ego + (R_ego - np.eye(3)) @ mount)
    # Scene points rotate opposite to the vehicle.
    R_cam = M @ R_ego.T @ M.T
    normal = M @ np.array([0.0, 0.0, -1.0])
    return R_cam, R_cam @ t_cam, normal, float(ipm_args(calib)["height_m"])


def _agreement(geometry_box: tuple[float, ...], tracker_box: list[float]) -> float:
    a = np.asarray([list(geometry_box)], dtype=float)
    b = np.asarray([list(tracker_box)], dtype=float)
    return float(box_iou_matrix(a, b)[0, 0])


async def propagate_frame(db: AsyncSession, *, from_frame_id: str, to_frame_id: str,
                          run_id: str | None = None, agree_iou: float = AGREE_IOU,
                          dry_run: bool = True) -> dict[str, Any]:
    """Propagate every eligible label from one frame to the next, or refuse and say why.

    `dry_run` defaults True: this writes labels into the corpus, and on the one corpus it has been measured
    against a copy did better (docs/MEASUREMENT.md, row 5), so writing is a person's decision.
    """
    src = await db.get(Frame, UUID(from_frame_id))
    dst = await db.get(Frame, UUID(to_frame_id))
    if src is None or dst is None:
        return {"measured": False, "reason": "one of the frames does not exist"}
    if src.session_id != dst.session_id:
        return {"measured": False,
                "reason": "the two frames are from different sessions, so there is no ego motion "
                          "between them"}

    motion = await ego_transform(db, src, dst)
    if not motion["measured"]:
        # The guard: an uncompensated copy assumes the camera stood still, which is wrong for every static
        # object and most wrong for the roadside furniture propagation should be best at.
        log.info("ego_propagate.refused", from_frame=from_frame_id, to_frame=to_frame_id,
                 reason=motion["reason"])
        return {"measured": False, "from_frame_id": from_frame_id, "to_frame_id": to_frame_id,
                "reason": motion["reason"], "n_propagated": 0, "n_conflicts": 0}

    from packs.registry import default_pack_id, get_pack
    from services.autolabel.ontology import get_ontology
    from services.calibration.resolve import resolve_calibration

    spec = get_pack(default_pack_id()).motion_models
    onto = get_ontology()
    if spec is None:
        return {"measured": False, "reason": "the pack defines no motion models, so no box can be warped"}
    calib = await resolve_calibration(src.session_id, src.cam_id, int(src.width), int(src.height))
    R_cam, t_cam, normal, height = camera_motion(motion, calib)
    H = ground_homography(calib.K(), R_cam, t_cam, normal=normal, height=height)
    src_objs = (await db.execute(select(Object).where(Object.frame_id == src.frame_id))).scalars().all()
    dst_objs = (await db.execute(select(Object).where(Object.frame_id == dst.frame_id))).scalars().all()
    by_track = {o.track_id: o for o in dst_objs if o.track_id is not None}

    written, conflicts, refused = 0, 0, 0
    for o in src_objs:
        try:
            model = spec.model_for(onto.by_id(o.class_id).name)
        except Exception:  # noqa: BLE001
            model = "moving"
        w = warp_box(list(o.bbox), H, width=dst.width, height=dst.height, motion_model=model)
        if not w.measured:
            refused += 1
            continue
        tracked = by_track.get(o.track_id)
        if tracked is not None:
            iou = _agreement(w.box, list(tracked.bbox))
            if iou < agree_iou:
                conflicts += 1
                if not dry_run:
                    db.add(PropagationConflict(
                        session_id=src.session_id, from_frame_id=src.frame_id,
                        to_frame_id=dst.frame_id, object_id=o.object_id, class_id=o.class_id,
                        motion_model=model, geometry_box=list(w.box),
                        tracker_box=list(tracked.bbox), iou=round(iou, 4), tolerance=agree_iou,
                        reason="ego geometry and the tracker propose different boxes; neither written"))
                continue
        if not dry_run:
            db.add(Object(frame_id=dst.frame_id, track_id=o.track_id, class_id=o.class_id,
                          bbox=list(w.box), conf=float(o.conf), source="propagated", state="review",
                          provenance={"created_by": "ego-propagation",
                                      "derived_from_run": run_id,
                                      "from_object_id": str(o.object_id),
                                      "motion_model": model, "shrink": w.shrink}))
        written += 1
    if not dry_run:
        await db.commit()

    log.info("ego_propagate.done", from_frame=from_frame_id, to_frame=to_frame_id,
             propagated=written, conflicts=conflicts, refused=refused, dry_run=dry_run)
    return {"measured": True, "from_frame_id": from_frame_id, "to_frame_id": to_frame_id,
            "n_propagated": written, "n_conflicts": conflicts, "n_refused": refused,
            "dry_run": dry_run}


async def coverage_report(db: AsyncSession) -> dict[str, Any]:
    """How much of the corpus could be propagated at all, read from the data rather than assumed.

    A frame is propagatable when it and the next frame from the same camera both have a measured pose,
    because only then does the guard in `propagate_frame` pass.
    """
    from sqlalchemy import func, text

    total = (await db.execute(select(func.count()).select_from(Frame))).scalar_one()
    with_gnss = (await db.execute(select(func.count()).select_from(Frame).where(
        Frame.gnss.is_not(None)))).scalar_one()
    with_speed = (await db.execute(select(func.count()).select_from(Frame).where(
        Frame.ego_speed.is_not(None)))).scalar_one()
    pairs = (await db.execute(text("""
        with f as (select session_id, ts_ns,
                          lead(ts_ns) over (partition by session_id, cam_id order by ts_ns) as nts
                   from frame)
        select count(*) from f
        join ego_pose a on a.session_id = f.session_id and a.ts_ns = f.ts_ns and a.measured
        join ego_pose b on b.session_id = f.session_id and b.ts_ns = f.nts and b.measured
    """))).scalar_one()
    return {"n_frames": int(total), "n_with_gnss": int(with_gnss), "n_with_ego_speed": int(with_speed),
            "pct_with_gnss": round(100.0 * with_gnss / total, 4) if total else 0.0,
            "n_propagatable_pairs": int(pairs),
            "propagatable": bool(pairs),
            "reason": None if pairs else
                      ("a rigid camera transform needs a measured pose on both frames of a pair, and no "
                       "pair of consecutive frames has one; GNSS (or a dashcam's burned-in stamp) is "
                       "what provides it")}


__all__ = ["camera_motion", "ego_transform", "propagate_frame", "coverage_report", "PoseUnavailable", "AGREE_IOU"]
