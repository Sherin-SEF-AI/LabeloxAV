"""Milestone G: the static / dynamic separation workflow. A parked car, a pole, a sign does not move across
the clip, so it should be labeled once and that label applied to every frame; a moving object must be
reviewed per frame. The motion is already classified per 3D track (track3d/dynamics.py -> dynamic_state),
so this routes each track to the right labeling strategy rather than re-deriving motion. A track is static
only when every observed dynamic_state is parked; any moving / turning / braking / stopped (it was moving at
some point, so its box changes) makes it dynamic.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("static_dynamic")

_STATIC_STATES = {"parked"}


def classify_track_motion(states: list) -> dict:
    """Route a track by its observed dynamic_state values. all parked -> static, label once. any motion ->
    dynamic, label per frame. nothing observed -> unknown, per frame (do not assume static)."""
    observed = {s for s in states if s}
    if not observed:
        return {"motion": "unknown", "label_strategy": "per_frame"}
    if observed <= _STATIC_STATES:
        return {"motion": "static", "label_strategy": "once"}
    return {"motion": "dynamic", "label_strategy": "per_frame"}


async def session_static_dynamic_split(session_id) -> dict:
    """Partition a session's 3D tracks into static (label once) and dynamic (label per frame)."""
    from sqlalchemy import select

    from db.models import Track3D
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Track3D.track_3d_id, Track3D.track_id, Track3D.dynamic_state)
            .where(Track3D.session_id == session_id))).all()
    static, dynamic = [], []
    for t3d_id, track_id, state in rows:
        c = classify_track_motion([state])
        entry = {"track_3d_id": str(t3d_id), "track_id": str(track_id) if track_id else None,
                 "dynamic_state": state, **c}
        (static if c["motion"] == "static" else dynamic).append(entry)
    log.info("static_dynamic.split", session=str(session_id), static=len(static), dynamic=len(dynamic))
    return {"session_id": str(session_id), "static": static, "dynamic": dynamic,
            "counts": {"static": len(static), "dynamic": len(dynamic), "total": len(rows)}}
