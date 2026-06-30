"""Milestone G: temporal attribute transitions. An object's attribute can change across a track (a traffic
signal turning red, brake lights on then off, an indicator blinking). Per-frame attrs already carry the
value; this collapses a track's per-frame values into contiguous segments, so the timeline shows the
transitions and an annotator edits a value once over a span instead of frame by frame.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("temporal_attributes")


def attribute_segments(objects: list[dict], key: str) -> list[dict]:
    """objects: [{ts_ns, attrs}] sorted by ts_ns. Contiguous runs of attrs[key] become segments
    {value, t_start_ns, t_end_ns}. A frame where the attribute is absent (None) breaks the run, so a gap in
    annotation is not bridged into a false transition."""
    segs: list[dict] = []
    cur: dict | None = None
    for o in objects:
        v = (o.get("attrs") or {}).get(key)
        if cur is None or cur["value"] != v:
            if cur is not None and cur["value"] is not None:
                segs.append(cur)
            cur = {"value": v, "t_start_ns": o["ts_ns"], "t_end_ns": o["ts_ns"]}
        else:
            cur["t_end_ns"] = o["ts_ns"]
    if cur is not None and cur["value"] is not None:
        segs.append(cur)
    return segs


async def track_attribute_timeline(track_id, key: str) -> dict:
    """The transition timeline of one attribute across a track: the contiguous value segments and the count
    of transitions between them."""
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Frame.ts_ns, Object.attrs).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns))).all()
    objects = [{"ts_ns": int(ts), "attrs": attrs or {}} for ts, attrs in rows]
    segs = attribute_segments(objects, key)
    return {"track_id": str(track_id), "key": key, "segments": segs,
            "transitions": max(0, len(segs) - 1)}
