"""Error-candidate queue (M4.1): run the three detectors, rank and persist candidates, and route them to
human review. Confirming an error feeds it back three ways: it writes a correction (Review), it remains an
error-prone signal the M4.0 selector reads, and the retrain loop counts confirmed errors as new signal.
This is QA on the gate itself."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns
from db.models import ErrorCandidate, Object, Review
from services.errordetect.confident import detect_confident_learning
from services.errordetect.consistency import detect_consistency
from services.errordetect.critic_detector import detect_critic
from services.errordetect.embedding_outlier import detect_embedding_outliers
from services.errordetect.near_dup import detect_near_dup_inconsistent
from services.errordetect.policy import detect_policy_violations

log = get_logger("ed_queue")


async def run_detection(db: AsyncSession, session_id: str | None = None, kinds: list[str] | None = None) -> dict:
    """Run the requested detectors and persist ranked error candidates. Idempotent: clears prior pending
    candidates of the run kinds before reinserting (confirmed/dismissed verdicts are preserved)."""
    kinds = kinds or ["confident_learning", "embedding_outlier", "track_inconsistent", "cross_cam_inconsistent", "critic_flag", "near_dup_inconsistent", "policy_violation"]
    found: list[dict] = []
    if "confident_learning" in kinds:
        found += await detect_confident_learning(db, session_id)
    if "embedding_outlier" in kinds:
        found += await detect_embedding_outliers(db, session_id)
    if {"track_inconsistent", "cross_cam_inconsistent"} & set(kinds):
        found += [c for c in await detect_consistency(db, session_id) if c["kind"] in kinds]
    if "critic_flag" in kinds:
        found += await detect_critic(db, session_id)
    if "near_dup_inconsistent" in kinds:
        found += await detect_near_dup_inconsistent(db, session_id)
    if "policy_violation" in kinds:
        found += await detect_policy_violations(db, session_id)

    # keep the strongest candidate per (object, kind)
    best: dict = {}
    for c in found:
        key = (c["object_id"], c["kind"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c

    await db.execute(delete(ErrorCandidate).where(ErrorCandidate.kind.in_(kinds), ErrorCandidate.status == "pending"))
    for c in best.values():
        db.add(ErrorCandidate(object_id=UUID(c["object_id"]), kind=c["kind"], score=c["score"],
                              proposed_label=c["proposed_label"], detail=c.get("detail", {}), status="pending"))
    await db.commit()

    by_kind: dict = {}
    for c in best.values():
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    log.info("ed.run", total=len(best), by_kind=by_kind, session_id=session_id)
    return {"persisted": len(best), "by_kind": by_kind, "scope": session_id or "corpus"}


async def list_candidates(db: AsyncSession, status: str = "pending", limit: int = 100) -> list[dict]:
    rows = (await db.execute(
        select(ErrorCandidate).where(ErrorCandidate.status == status)
        .order_by(ErrorCandidate.score.desc()).limit(limit))).scalars().all()
    return [{"candidate_id": str(c.candidate_id), "object_id": str(c.object_id), "kind": c.kind,
             "score": c.score, "proposed_label": c.proposed_label, "detail": c.detail, "status": c.status}
            for c in rows]


async def confirm_error(db: AsyncSession, candidate_id: str, apply_proposed: bool = True,
                        reviewer: str = "error-detect", user_id: str | None = None) -> dict:
    """Confirm a candidate as a real error. Applies the proposed class fix when present (a correction the
    retrain consumes), otherwise routes the object back to human annotation."""
    c = await db.get(ErrorCandidate, UUID(candidate_id))
    if c is None:
        return {"error": "candidate not found"}
    obj = await db.get(Object, c.object_id)
    if obj is None:
        return {"error": "object not found"}

    before = {"class_id": obj.class_id, "state": obj.state, "source": obj.source}
    if apply_proposed and c.proposed_label and c.proposed_label.get("class_id") is not None:
        obj.class_id = int(c.proposed_label["class_id"])
        obj.state, obj.source = "accepted", "human"
        action = "error_fix"
    else:
        obj.state = "review"  # re-enter the human queue (no confident class proposal)
        action = "error_flag"
    after = {"class_id": obj.class_id, "state": obj.state, "source": obj.source}
    db.add(Review(object_id=obj.object_id, reviewer=reviewer, user_id=UUID(user_id) if user_id else None,
                  action=action, before=before, after=after, time_spent_ms=0, ts_ns=now_ns()))
    c.status = "confirmed_error"
    await db.commit()
    log.info("ed.confirmed", candidate_id=candidate_id, kind=c.kind, action=action)
    return {"candidate_id": candidate_id, "status": "confirmed_error", "action": action, "before": before, "after": after}


async def dismiss_error(db: AsyncSession, candidate_id: str) -> dict:
    c = await db.get(ErrorCandidate, UUID(candidate_id))
    if c is None:
        return {"error": "candidate not found"}
    c.status = "dismissed"
    await db.commit()
    return {"candidate_id": candidate_id, "status": "dismissed"}


async def summary(db: AsyncSession) -> dict:
    rows = (await db.execute(
        select(ErrorCandidate.status, func.count()).group_by(ErrorCandidate.status))).all()
    return {status: int(n) for status, n in rows}
