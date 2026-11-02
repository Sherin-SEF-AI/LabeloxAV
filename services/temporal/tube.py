"""Tube scores for stored tracks: how much each behaves like one real object over its span.

core/accel/tube_score.py does the scoring from boxes and presence flags. This builds those from the corpus:
every frame of the track's camera between its first and last box, the box where the track has one and a
gap where it does not, and the class on each. A track shorter than the scorer's minimum gets None, which is
"not measured" rather than a low score.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.tube_score import tube_score
from db.models import Frame, Object, Track


async def score_track(db: AsyncSession, track_id: uuid.UUID) -> float | None:
    rows = (await db.execute(
        select(Frame.frame_id, Frame.session_id, Frame.cam_id, Frame.ts_ns, Object.bbox, Object.class_id)
        .join(Object, Object.frame_id == Frame.frame_id).where(Object.track_id == track_id)
        .order_by(Frame.ts_ns))).all()
    if not rows:
        return None
    sid, cam, lo, hi = rows[0][1], rows[0][2], rows[0][3], rows[-1][3]
    span = (await db.execute(select(Frame.frame_id).where(
        Frame.session_id == sid, Frame.cam_id == cam, Frame.ts_ns >= lo, Frame.ts_ns <= hi)
        .order_by(Frame.ts_ns))).scalars().all()
    at = {fid: (bbox, cid) for fid, _s, _c, _t, bbox, cid in rows}
    boxes = [at[f][0] if f in at else [0.0, 0.0, 0.0, 0.0] for f in span]
    ts = tube_score(boxes, [f in at for f in span], [at[f][1] if f in at else -1 for f in span])
    return float(ts.score) if ts.measured and ts.score is not None else None


async def score_session_tracks(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """Score and store every track of a session. Returns how many were measured."""
    tids = (await db.execute(select(Track.track_id).where(Track.session_id == session_id))).scalars().all()
    measured = 0
    for tid in tids:
        score = await score_track(db, tid)
        await db.execute(update(Track).where(Track.track_id == tid).values(tube_score=score))
        measured += score is not None
    return {"tracks": len(tids), "measured": measured}


__all__ = ["score_session_tracks", "score_track"]
