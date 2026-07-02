"""Consistency-critic error detector: bring the annotation agent's self-consistency checks into the error
queue. detect_consistency already covers track class-flips; this adds the checks it does not -- a ground
object above the horizon (geometric), a pedestrian doing 60 km/h (motion), a rider with no two-wheeler
(relationship), and a vehicle box empty of LiDAR (cross-modal). Each flagged object becomes a ranked
ErrorCandidate the fix queue surfaces alongside the confident-learning and embedding-outlier detectors.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("ed.critic")

# severity per check -> the candidate's score (temporal is intentionally omitted: detect_consistency owns it)
_SEVERITY = {"geometric": 0.8, "cross_modal": 0.75, "motion": 0.7, "relationship": 0.6}


async def detect_critic(db: AsyncSession, session_id: str | None = None, *, limit_frames: int | None = None) -> list[dict]:
    from services.agent.critic import critique_frame
    from services.agent.frame_agent import _build_context, _load_objects

    q = select(distinct(Object.frame_id)).where(Object.source != "human")
    if session_id:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == UUID(session_id))
    if limit_frames:
        q = q.limit(limit_frames)
    frame_ids = list((await db.execute(q)).scalars().all())

    out: list[dict] = []
    for fid in frame_ids:
        frame = await db.get(Frame, fid)
        if frame is None:
            continue
        objs = await _load_objects(db, fid)
        if not objs:
            continue
        ctx = await _build_context(db, frame, objs)
        verdicts = critique_frame(ctx)
        for o in objs:
            v = verdicts.get(str(o.object_id))
            if v is None:
                continue
            flags = [c for c, st in v.checks.items() if st == "flag" and c in _SEVERITY]
            if not flags:
                continue
            score = max(_SEVERITY[c] for c in flags)
            out.append({"object_id": str(o.object_id), "kind": "critic_flag", "score": round(score, 4),
                        "proposed_label": None, "detail": {"checks": flags, "reasons": v.reasons}})
    log.info("ed.critic.done", frames=len(frame_ids), flagged=len(out), scope=session_id or "corpus")
    return out
