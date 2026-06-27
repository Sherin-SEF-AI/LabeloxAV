"""Keyframe selection + edit-propagation (M2.5): mark a frame as a human keyframe, and when a keyframe is
fixed re-interpolate only the segments adjacent to it (between its neighbor keyframes), not the whole
track."""

from __future__ import annotations

from uuid import UUID

from db.models import Frame, Object
from db.session import get_sessionmaker
from services.temporal.interpolate import _keyframes, interpolate_track_keyframed


async def mark_keyframe(object_id: UUID, value: bool = True) -> dict:
    maker = get_sessionmaker()
    async with maker() as db:
        o = await db.get(Object, object_id)
        if o is None:
            return {"ok": False, "reason": "object not found"}
        o.is_keyframe = value
        if value:
            o.source, o.state = "human", "accepted"
        await db.commit()
        return {"object_id": str(object_id), "is_keyframe": value,
                "track_id": str(o.track_id) if o.track_id else None}


async def reinterpolate_segment(object_id: UUID, method: str = "linear") -> dict:
    """Re-interpolate only the two segments adjacent to the edited keyframe."""
    maker = get_sessionmaker()
    async with maker() as db:
        o = await db.get(Object, object_id)
        if o is None or o.track_id is None:
            return {"created": 0, "reason": "object has no track"}
        track_id = o.track_id
        frame = await db.get(Frame, o.frame_id)
        ts = frame.ts_ns
        kf_ts = sorted(t for _, t in await _keyframes(db, track_id))

    lo = max([t for t in kf_ts if t < ts], default=None)
    hi = min([t for t in kf_ts if t > ts], default=None)
    created = 0
    if lo is not None:
        created += (await interpolate_track_keyframed(track_id, method, lo_ts=lo, hi_ts=ts))["created"]
    if hi is not None:
        created += (await interpolate_track_keyframed(track_id, method, lo_ts=ts, hi_ts=hi))["created"]
    return {"track_id": str(track_id), "created": created, "around_ts": ts}
