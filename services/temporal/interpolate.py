"""Keyframe interpolation along a track (M2.5). Between human keyframes, fill the frames with linear or
cubic boxes (and SAM 3.1 mask propagation on the pod), marking them source=interpolated + interp_source so
provenance shows they are machine-filled. Builds on the box-interpolation geometry from propagate.py and
the BoT-SORT tracks from M2.0.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import delete, or_, select

from core.logging import get_logger
from db.models import Frame, Object, Track
from db.session import get_sessionmaker

log = get_logger("interpolate")


async def _keyframes(db, track_id: UUID):
    """Anchor objects on a track: human-verified or explicitly-marked keyframes, ordered by time."""
    rows = (await db.execute(
        select(Object, Frame.ts_ns).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id == track_id, or_(Object.is_keyframe.is_(True), Object.source == "human"))
        .order_by(Frame.ts_ns))).all()
    return rows


async def interpolate_track_keyframed(track_id: UUID, method: str = "linear", lo_ts: int | None = None, hi_ts: int | None = None) -> dict:
    """Fill frames between keyframes with interpolated boxes. If lo_ts/hi_ts are given, only that segment
    is (re)interpolated (edit-propagation); otherwise the whole track between first and last keyframe."""
    maker = get_sessionmaker()
    async with maker() as db:
        tr = await db.get(Track, track_id)
        if tr is None:
            return {"created": 0, "reason": "track not found"}
        anchors = await _keyframes(db, track_id)
        if len(anchors) < 2:
            return {"created": 0, "reason": "need at least 2 keyframes (mark human-verified frames)"}

        kf_ts = [ts for _, ts in anchors]
        kf_box = np.asarray([list(o.bbox) for o, _ in anchors], dtype=float)
        class_id = anchors[0][0].class_id
        a, b = (lo_ts if lo_ts is not None else kf_ts[0]), (hi_ts if hi_ts is not None else kf_ts[-1])

        frames = (await db.execute(
            select(Frame.frame_id, Frame.ts_ns)
            .where(Frame.session_id == tr.session_id, Frame.ts_ns > a, Frame.ts_ns < b)
            .order_by(Frame.ts_ns))).all()
        # clear existing machine-filled boxes on this track in the segment (idempotent re-interpolation)
        seg_fids = [fid for fid, _ in frames]
        if seg_fids:
            await db.execute(delete(Object).where(
                Object.track_id == track_id, Object.source == "interpolated", Object.frame_id.in_(seg_fids)))

        use_cubic = method == "cubic" and len(kf_ts) >= 3
        if use_cubic:
            from scipy.interpolate import interp1d
            fns = [interp1d(kf_ts, kf_box[:, i], kind="cubic", fill_value="extrapolate") for i in range(4)]
            interp = lambda ts: [float(fns[i](ts)) for i in range(4)]  # noqa: E731
            src = "cubic"
        else:
            interp = lambda ts: [float(np.interp(ts, kf_ts, kf_box[:, i])) for i in range(4)]  # noqa: E731
            src = "linear"

        kf_set = set(kf_ts)
        created = 0
        for fid, ts in frames:
            if ts in kf_set:
                continue
            db.add(Object(frame_id=fid, track_id=track_id, class_id=class_id, bbox=interp(ts),
                          conf=0.5, source="interpolated", state="annotate", interp_source=src,
                          provenance={"method": "interpolate", "interp_source": src}))
            created += 1
        await db.commit()

    out = {"track_id": str(track_id), "created": created, "method": src, "keyframes": len(kf_ts)}
    log.info("interpolate.done", **out)
    return out
