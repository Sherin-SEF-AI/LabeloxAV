"""Track a session's 3D objects across frames and persist track_3d. Each object_3d becomes a detection
carrying its 2D object's track_id (the link to the M2.0 track) and the frame's ego speed; the tracker
associates them by 3D IoU into 3D tracks. For each confirmed track we write a track_3d (linked to the 2D
track, with a trajectory and a dynamic state), assign object_3d.track_3d_id, and interpolate any gaps between
human keyframes. Raw is never mutated; only label records are written.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select, update

from core.logging import get_logger
from db.models import Frame, Object, Object3D, PointCloud, Track3D
from db.session import get_sessionmaker
from services.lidar.track3d.dynamics import classify_track
from services.lidar.track3d.tracker3d import Tracker3D

log = get_logger("lidar_track3d_run")


async def _gather(session_id: uuid.UUID) -> list[dict]:
    """Every machine 3D object in the session, with its cloud ts_ns, its 2D track link, and the ego speed."""
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object3D, PointCloud.ts_ns, Object.track_id, Frame.ego_speed)
            .join(PointCloud, Object3D.cloud_id == PointCloud.cloud_id)
            .outerjoin(Object, Object3D.object_id == Object.object_id)
            .outerjoin(Frame, Object3D.frame_id == Frame.frame_id)
            .where(PointCloud.session_id == session_id, Object3D.source != "human")
            .order_by(PointCloud.ts_ns))).all()
    dets = []
    for o, ts_ns, track_id_2d, ego_speed in rows:
        dets.append({"object_3d_id": o.object_3d_id, "center": o.center, "dims": o.dims, "yaw": o.yaw,
                     "class_id": o.class_id, "ts_ns": int(ts_ns), "ego_speed": ego_speed,
                     "track_id_2d": str(track_id_2d) if track_id_2d else None})
    return dets


def _valid_uuid(s: str | None) -> uuid.UUID | None:
    try:
        return uuid.UUID(s) if s else None
    except Exception:
        return None


async def track_session(session_id: uuid.UUID) -> dict:
    """Run the 3D tracker over the session and persist track_3d, linking each to its 2D track."""
    dets = await _gather(session_id)
    if not dets:
        return {"session_id": str(session_id), "tracks": 0, "reason": "no machine 3D objects"}

    # group detections by frame (shared ts_ns) and step the tracker in time order
    frames: dict[int, list[dict]] = {}
    for d in dets:
        frames.setdefault(d["ts_ns"], []).append(d)
    ordered = sorted(frames.items())

    tracker = Tracker3D()
    prev_ts = ordered[0][0]
    members: dict[int, list[dict]] = {}
    for ts_ns, group in ordered:
        dt = max((ts_ns - prev_ts) / 1e9, 1e-3)
        prev_ts = ts_ns
        assigns = tracker.step(group, dt=dt)
        for a in assigns:
            local = a["track_3d_local_id"]
            if local is not None:
                members.setdefault(local, []).append(group[a["detection_index"]])

    confirmed = tracker.confirmed()
    written = 0
    async with get_sessionmaker()() as db:
        for t in confirmed:
            mem = members.get(t.id, [])
            if not mem:
                continue
            ts_sorted = sorted(mem, key=lambda m: m["ts_ns"])
            samples = [{"ts_ns": m["ts_ns"], "center": m["center"], "yaw": m["yaw"],
                        "ego_speed": m["ego_speed"]} for m in ts_sorted]
            dyn = classify_track(samples)
            cls = max({m["class_id"] for m in mem}, key=lambda c: sum(1 for m in mem if m["class_id"] == c))
            trajectory = [{"ts_ns": m["ts_ns"], "center": m["center"], "yaw": m["yaw"]} for m in ts_sorted]
            tr = Track3D(track_id=_valid_uuid(t.linked_track_2d()), session_id=session_id, class_id=cls,
                         first_ts_ns=ts_sorted[0]["ts_ns"], last_ts_ns=ts_sorted[-1]["ts_ns"],
                         trajectory={"points": trajectory}, dynamic_state=dyn["state"])
            db.add(tr)
            await db.flush()
            ids = [m["object_3d_id"] for m in mem]
            await db.execute(update(Object3D).where(Object3D.object_3d_id.in_(ids))
                             .values(track_3d_id=tr.track_3d_id))
            written += 1
        await db.commit()
    log.info("lidar.track_session", session=str(session_id), detections=len(dets), tracks=written)
    return {"session_id": str(session_id), "detections": len(dets), "frames": len(ordered), "tracks": written}
