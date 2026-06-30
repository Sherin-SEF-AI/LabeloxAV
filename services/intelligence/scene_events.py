"""Milestone B (scene layer): turn the per-frame SigLIP2 scene classification into contiguous scene events
on the timeline. Adverse conditions (rain, fog, night, dusk, dawn) are segmented into runs and persisted as
scene-modality TimelineEvents (source=auto, unconfirmed), so a labeler can confirm the adverse window and
pair it with a clear-weather re-drive. Reuses the existing frame.scene, no new model.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("scene_events")

# the adverse values per scene axis (clear/day/etc. are not adverse)
_ADVERSE = {"weather": {"rain", "fog"}, "time_of_day": {"night", "dusk", "dawn"}}


def adverse_conditions(scene: dict | None) -> list[str]:
    """The adverse-condition kinds present in one frame's scene classification."""
    out = []
    for axis, bad in _ADVERSE.items():
        v = (scene or {}).get(axis)
        if v in bad:
            out.append(v)
    return out


def segment_scene_conditions(frames: list[tuple]) -> list[dict]:
    """frames: [(ts_ns, scene_dict)] sorted by ts_ns. Contiguous runs of each adverse condition become scene
    events {kind, t_start_ns, t_end_ns}; the end is the last frame the condition was present."""
    run_start: dict[str, int] = {}
    last_ts: dict[str, int] = {}
    prev: set[str] = set()
    segments: list[dict] = []
    for ts, scene in frames:
        conds = set(adverse_conditions(scene))
        for c in prev - conds:                       # a condition just ended
            segments.append({"kind": c, "t_start_ns": run_start[c], "t_end_ns": last_ts[c]})
            run_start.pop(c, None)
        for c in conds - prev:                       # a condition just started
            run_start[c] = ts
        for c in conds:
            last_ts[c] = ts
        prev = conds
    for c, t in run_start.items():                   # close runs open at the clip end
        segments.append({"kind": c, "t_start_ns": t, "t_end_ns": last_ts[c]})
    return sorted(segments, key=lambda s: s["t_start_ns"])


async def persist_scene_events(session_id) -> dict:
    """Segment a session's adverse-condition runs from frame.scene and persist them as unconfirmed scene
    events. Idempotent: clears prior auto scene events first."""
    from sqlalchemy import delete, select

    from db.models import Frame, TimelineEvent
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Frame.ts_ns, Frame.scene).where(
            Frame.session_id == session_id, Frame.scene.isnot(None)).order_by(Frame.ts_ns))).all()
        frames = [(int(ts), scene) for ts, scene in rows]
        segs = segment_scene_conditions(frames)
        await db.execute(delete(TimelineEvent).where(
            TimelineEvent.session_id == session_id, TimelineEvent.modality == "scene",
            TimelineEvent.source == "auto"))
        for s in segs:
            db.add(TimelineEvent(session_id=session_id, kind=s["kind"], modality="scene",
                                 t_start_ns=s["t_start_ns"], t_end_ns=s["t_end_ns"], payload={},
                                 source="auto", state="review",
                                 provenance={"detector": "scene_axes", "axis": "weather/time_of_day"}))
        await db.commit()
    log.info("scene.events", session=str(session_id), segments=len(segs))
    return {"session_id": str(session_id), "scene_events": len(segs), "all_unconfirmed": True}
