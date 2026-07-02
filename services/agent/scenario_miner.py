"""Rare-scenario + safety-event miner: the system finds what is worth labeling. It reads the derived
dynamics and inertial timeline to surface the safety-critical moments a dataset most needs -- near-misses
(an object with a low time-to-collision), high-risk interactions, and hard-brake / swerve inertial events --
and writes them to the ScenarioCandidate queue (kinds near_miss / high_risk / hard_brake), so they show up
in the existing scenarios/discovery review UI alongside the embedding-novelty rare scenes. Idempotent: it
clears prior pending safety candidates before reinserting, preserving confirmed/dismissed verdicts.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectDynamics, ScenarioCandidate, TimelineEvent

log = get_logger("agent.scenario_miner")

_SAFETY_KINDS = ["near_miss", "high_risk", "hard_brake"]
_BRAKE_TOKENS = ("brake", "hard_accel", "accel", "swerve", "jerk", "cut_in", "cutin")


async def mine_scenarios(db: AsyncSession, session_id: str | None = None, *, ttc_thresh: float = 2.5) -> dict:
    """Mine safety-critical scenarios into the ScenarioCandidate queue. Returns counts by kind + top items."""
    found: dict[tuple, dict] = {}  # (frame_id, kind) -> best candidate

    def _add(sid, fid, kind, score, tag):
        key = (str(fid) if fid else str(sid), kind)
        if key not in found or score > found[key]["score"]:
            found[key] = {"session_id": sid, "frame_id": fid, "kind": kind, "score": round(float(score), 3), "tag": tag}

    # near-miss: objects with a low time-to-collision
    q = (select(Object.frame_id, Frame.session_id, ObjectDynamics.ttc_s)
         .join(Object, Object.object_id == ObjectDynamics.object_id)
         .join(Frame, Frame.frame_id == Object.frame_id)
         .where(ObjectDynamics.ttc_s.isnot(None), ObjectDynamics.ttc_s < ttc_thresh))
    if session_id:
        q = q.where(Frame.session_id == UUID(session_id))
    for fid, sid, ttc in (await db.execute(q)).all():
        _add(sid, fid, "near_miss", (ttc_thresh - float(ttc)) / ttc_thresh, f"near-miss: TTC {float(ttc):.1f}s")

    # high-risk interactions
    rq = (select(Object.frame_id, Frame.session_id).select_from(ObjectDynamics)
          .join(Object, Object.object_id == ObjectDynamics.object_id)
          .join(Frame, Frame.frame_id == Object.frame_id)
          .where(ObjectDynamics.risk_level == "high"))
    if session_id:
        rq = rq.where(Frame.session_id == UUID(session_id))
    for fid, sid in (await db.execute(rq)).all():
        _add(sid, fid, "high_risk", 0.7, "high-risk interaction")

    # hard-brake / swerve inertial events (0 in a camera-only corpus; fires on IMU-equipped sessions)
    eq = select(TimelineEvent).where(TimelineEvent.modality == "imu")
    if session_id:
        eq = eq.where(TimelineEvent.session_id == UUID(session_id))
    for e in (await db.execute(eq)).scalars().all():
        if any(tok in str(e.kind).lower() for tok in _BRAKE_TOKENS):
            # bind to the nearest frame if one exists
            fid = (await db.execute(select(Frame.frame_id).where(Frame.session_id == e.session_id)
                   .order_by(Frame.ts_ns).limit(1))).scalar()
            _add(e.session_id, fid, "hard_brake", 0.65, f"inertial {e.kind}")

    # persist idempotently
    del_q = delete(ScenarioCandidate).where(ScenarioCandidate.kind.in_(_SAFETY_KINDS), ScenarioCandidate.state == "pending")
    if session_id:
        del_q = del_q.where(ScenarioCandidate.session_id == UUID(session_id))
    await db.execute(del_q)
    for c in found.values():
        db.add(ScenarioCandidate(session_id=c["session_id"], frame_id=c["frame_id"], kind=c["kind"],
                                 score=c["score"], state="pending", tag=c["tag"]))
    await db.commit()

    by_kind: dict = {}
    for c in found.values():
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    top = sorted(found.values(), key=lambda c: -c["score"])[:10]
    log.info("agent.scenario_miner", persisted=len(found), by_kind=by_kind, scope=session_id or "corpus")
    return {"persisted": len(found), "by_kind": by_kind,
            "top": [{"kind": c["kind"], "score": c["score"], "tag": c["tag"],
                     "frame_id": str(c["frame_id"]) if c["frame_id"] else None} for c in top]}
