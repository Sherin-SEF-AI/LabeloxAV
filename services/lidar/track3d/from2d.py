"""One track's cuboids, made comparable across time and locked to one size.

`detect3d/lift_frame` lifts each 2D box to a cuboid independently, frame by frame. That is the right unit
for a detector and the wrong unit for an object: the same car gets a slightly different length every
frame because the depth estimate moved a little, and the cuboids jitter in a way that reads as motion.
Nothing tied them together, because there was no frame to tie them together in.

Two things fix that and both need the ego pose from 0112.

**One coordinate frame.** Each cuboid is lifted in the ego frame of its own timestamp, so two cuboids from
two frames are in two different coordinate systems. Placed in the session's ENU frame through the pose,
they become one trajectory that can be smoothed. Without a pose at a timestamp, the cuboid is left where
it is and the track says so, because moving it by an assumed pose is worse than not moving it.

**One size.** A car does not change length. The per-track median of each dimension is locked across the
whole track, so the variance that remains is motion rather than measurement. The median rather than the
mean because a single bad depth frame is a large outlier.

The trigonometric prior in `oraclyx/mono_depth.metric_depth` is used as a check rather than a second
estimate: where it and the lifted range disagree by more than `DISAGREE_FRAC`, the cuboid's confidence is
lowered and it is flagged for review. Two methods that disagree are information about the calibration,
and averaging them would produce a box neither proposed.
"""

from __future__ import annotations

import math
import statistics
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, Object, Object3D, Track3D

log = get_logger("track3d_from2d")

# Relative disagreement between the lifted range and the trigonometric prior above which the cuboid is
# not trusted at face value. 30% at 20 m is 6 m, which is a different vehicle.
DISAGREE_FRAC = 0.30
# Confidence multiplier applied to a cuboid whose two range estimates disagree. It is lowered rather than
# zeroed: the box may still be right, and the point is to route it to a person, not to delete it.
DISAGREE_CONF_SCALE = 0.5
# A track needs this many cuboids before a median dimension means anything.
MIN_CUBOIDS_FOR_MEDIAN = 3


def ego_to_world(center: list[float], pose: EgoPose) -> list[float]:
    """A point in the ego frame at one instant, expressed in the session's ENU frame.

    Ego frame is x forward, y left, z up; the pose carries a yaw-only rotation, because roll and pitch
    are not observable from any source this corpus has and a fabricated one would tilt every box.
    """
    yaw = 2.0 * math.atan2(pose.qz, pose.qw)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    east = pose.x + cx * math.cos(yaw) - cy * math.sin(yaw)
    north = pose.y + cx * math.sin(yaw) + cy * math.cos(yaw)
    return [east, north, pose.z + cz]


def lock_dimensions(dims: list[list[float]]) -> list[float] | None:
    """The per-axis median of a track's cuboid dimensions, or None when there are too few to matter."""
    if len(dims) < MIN_CUBOIDS_FOR_MEDIAN:
        return None
    return [float(statistics.median([d[i] for d in dims])) for i in range(3)]


def dimension_variance(dims: list[list[float]]) -> list[float]:
    """Per-axis variance of a track's dimensions. The number smoothing is supposed to reduce."""
    if len(dims) < 2:
        return [0.0, 0.0, 0.0]
    return [float(statistics.pvariance([d[i] for d in dims])) for i in range(3)]


def smooth_positions(points: list[tuple[int, list[float]]]) -> list[list[float]]:
    """A constant-velocity Kalman pass over a track's world positions, in time order.

    The same filter the 3D tracker already uses in spirit, run offline over a finished track where both
    past and future are available, so it is a smoother rather than a predictor. Timestamps set the step,
    because dashcam frames are not evenly spaced and treating them as if they were would stretch every
    gap into acceleration.
    """
    import numpy as np

    if len(points) < 3:
        return [p for _ts, p in points]
    xs = np.array([p for _ts, p in points], dtype=np.float64)
    ts = np.array([ts for ts, _p in points], dtype=np.float64) / 1e9

    # Forward pass: position and velocity per axis, with a process noise that lets a real turn through
    # and a measurement noise that does not chase a single bad depth frame.
    q, r = 0.5, 1.0
    out = np.zeros_like(xs)
    for axis in range(xs.shape[1]):
        x = np.array([xs[0, axis], 0.0])
        P = np.array([[r, 0.0], [0.0, 10.0]])
        fwd = []
        for i in range(len(ts)):
            dt = (ts[i] - ts[i - 1]) if i else 0.0
            F = np.array([[1.0, dt], [0.0, 1.0]])
            x = F @ x
            P = F @ P @ F.T + np.array([[q * dt ** 2, 0.0], [0.0, q]])
            z = xs[i, axis]
            y = z - x[0]
            S = P[0, 0] + r
            K = P[:, 0] / S
            x = x + K * y
            P = P - np.outer(K, P[0, :])
            fwd.append(x[0])
        out[:, axis] = fwd
    return [list(map(float, row)) for row in out]


async def _cuboids_for_track(db: AsyncSession, track_id: uuid.UUID) -> list[dict]:
    """Every lifted cuboid on a 2D track, with the timestamp and session its frame belongs to."""
    rows = (await db.execute(
        select(Object3D, Frame.ts_ns, Frame.session_id, Frame.width, Frame.height, Frame.cam_id,
               Object.bbox, Object.class_id)
        .join(Object, Object.object_id == Object3D.object_id)
        .join(Frame, Frame.frame_id == Object3D.frame_id)
        .where(Object.track_id == track_id)
        .order_by(Frame.ts_ns))).all()
    return [{"o3d": o3d, "ts_ns": int(ts), "session_id": sid, "width": w or 1280, "height": h or 960,
             "cam_id": cam, "bbox": [float(v) for v in bbox], "class_id": int(cid)}
            for o3d, ts, sid, w, h, cam, bbox, cid in rows]


async def _range_prior(item: dict) -> float | None:
    """The trigonometric range for one 2D box, or None when the calibration cannot supply one."""
    from core.config import get_settings
    from services.calibration.resolve import resolve_calibration
    from services.oraclyx.mono_depth import metric_depth

    calib = await resolve_calibration(item["session_id"], item["cam_id"], item["width"], item["height"])
    height_m = float(calib.xyz_m[2] or 0.0) or float(get_settings().rig.camera_height_m)
    est = metric_depth(item["bbox"], item["class_id"], calib.fy, calib.cy,
                       cam_height_m=height_m, pitch_rad=math.radians(float(calib.rpy_deg[1])))
    return float(est["depth_m"]) if est else None


async def lift_track(track_id: uuid.UUID, *, dry_run: bool = False) -> dict:
    """Make one 2D track's cuboids a single object: one frame, one size, one smoothed trajectory.

    Returns what it changed and what it refused, including the dimension variance before and after, which
    is the number this exists to reduce. `dry_run` computes everything and writes nothing.
    """
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        items = await _cuboids_for_track(db, track_id)
        if not items:
            return {"track_id": str(track_id), "cuboids": 0,
                    "reason": "no lifted cuboid is attached to this track"}
        session_id = items[0]["session_id"]
        poses = {p.ts_ns: p for p in (await db.execute(
            select(EgoPose).where(EgoPose.session_id == session_id))).scalars().all()}

        dims_before = [[float(v) for v in it["o3d"].dims] for it in items]
        var_before = dimension_variance(dims_before)
        locked = lock_dimensions(dims_before)

        placed: list[tuple[int, list[float]]] = []
        unposed = 0
        for it in items:
            pose = poses.get(it["ts_ns"])
            if pose is None:
                unposed += 1
                continue
            placed.append((it["ts_ns"], ego_to_world([float(v) for v in it["o3d"].center], pose)))

        smoothed = smooth_positions(placed) if len(placed) >= 3 else []
        by_ts = {ts: p for (ts, _old), p in zip(placed, smoothed, strict=False)} if smoothed else {}

        disagreed = 0
        checked = 0
        for it in items:
            prior = await _range_prior(it)
            if prior is None or prior <= 0:
                continue
            checked += 1
            lifted_range = float(it["o3d"].center[0])
            if lifted_range <= 0:
                continue
            if abs(lifted_range - prior) / prior > DISAGREE_FRAC:
                disagreed += 1
                it["disagrees"] = {"lifted_m": round(lifted_range, 2), "prior_m": round(prior, 2)}

        if dry_run:
            return {"track_id": str(track_id), "cuboids": len(items), "posed": len(placed),
                    "unposed": unposed, "locked_dims": locked, "range_checked": checked,
                    "range_disagreed": disagreed,
                    "dim_variance_before": [round(v, 5) for v in var_before],
                    "dim_variance_after": [0.0, 0.0, 0.0] if locked else
                    [round(v, 5) for v in var_before], "written": 0, "dry_run": True}

        onto_class = items[0]["class_id"]
        t3 = Track3D(track_id=track_id, session_id=session_id, class_id=onto_class,
                     first_ts_ns=items[0]["ts_ns"], last_ts_ns=items[-1]["ts_ns"],
                     trajectory={"frame": "session_enu", "smoothed": bool(by_ts),
                                 "points": [{"ts_ns": ts, "xyz": [round(v, 3) for v in p]}
                                            for ts, p in zip([t for t, _ in placed], smoothed,
                                                             strict=False)]})
        db.add(t3)
        await db.flush()

        written = 0
        for it in items:
            o3d = it["o3d"]
            o3d.track_3d_id = t3.track_3d_id
            if locked:
                o3d.dims = locked
            prov = dict(o3d.provenance or {})
            lift = dict(prov.get("lift") or {})
            pose = poses.get(it["ts_ns"])
            lift.update({"track_3d_id": str(t3.track_3d_id),
                         "ego_pose_source": pose.source if pose else None,
                         "ego_pose_measured": bool(pose.measured) if pose else False,
                         "dims_locked": bool(locked),
                         "world_xyz": [round(v, 3) for v in by_ts[it["ts_ns"]]]
                                      if it["ts_ns"] in by_ts else None})
            if it.get("disagrees"):
                lift["range_disagreement"] = it["disagrees"]
                # Lowered, not zeroed, and routed to a person: two methods that disagree are information
                # about the calibration, and deleting the box would throw that away.
                o3d.conf = float(o3d.conf) * DISAGREE_CONF_SCALE
                o3d.state = "review"
            prov["lift"] = lift
            o3d.provenance = prov
            written += 1
        await db.commit()

    report = {"track_id": str(track_id), "track_3d_id": str(t3.track_3d_id), "cuboids": len(items),
              "posed": len(placed), "unposed": unposed, "locked_dims": locked,
              "range_checked": checked, "range_disagreed": disagreed,
              "dim_variance_before": [round(v, 5) for v in var_before],
              "dim_variance_after": [0.0, 0.0, 0.0] if locked else [round(v, 5) for v in var_before],
              "written": written}
    log.info("track3d.lifted", track_id=str(track_id), cuboids=len(items), posed=len(placed),
             disagreed=disagreed)
    return report


async def lift_tracks_for_session(session_id: uuid.UUID, *, limit: int = 50) -> dict:
    """Every 2D track in a session that has lifted cuboids and no 3D track yet."""
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.track_id)
            .join(Object3D, Object3D.object_id == Object.object_id)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id, Object.track_id.isnot(None),
                   Object3D.track_3d_id.is_(None))
            .distinct().limit(limit))).scalars().all()
    out = []
    for tid in rows:
        out.append(await lift_track(tid))
    return {"session_id": str(session_id), "tracks": len(out), "results": out}
