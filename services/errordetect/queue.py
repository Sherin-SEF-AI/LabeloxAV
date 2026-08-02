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

# Below this many verdicts a detector's precision interval spans most of [0, 1], which is not a measurement.
# Reported as unmeasured rather than shown beside detectors that have been properly judged, so a detector
# with three verdicts cannot be quietly ranked against one with three hundred.
MIN_VERDICTS_FOR_PRECISION = 20


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


async def list_candidates(db: AsyncSession, status: str = "pending", limit: int = 100,
                          kind: str | None = None) -> list[dict]:
    """The ranked queue, optionally restricted to one detector.

    The restriction is not a convenience filter. Ranking is across detectors by score, and the detectors do
    not emit commensurable scores, so a mixed page is largely whichever detector produces the biggest
    numbers. It is also the only way to judge a single detector, which is what the verdicts exist for.
    """
    q = select(ErrorCandidate).where(ErrorCandidate.status == status)
    if kind:
        q = q.where(ErrorCandidate.kind == kind)
    rows = (await db.execute(q.order_by(ErrorCandidate.score.desc()).limit(limit))).scalars().all()
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

    before = {"class_id": obj.class_id, "state": obj.state, "source": obj.source,
              "conf": obj.conf, "provenance": dict(obj.provenance or {})}
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


async def dismiss_error(db: AsyncSession, candidate_id: str, *, user_id: str | None = None,
                        note: str | None = None) -> dict:
    """Dismiss a candidate: the detector was wrong about this object.

    Records who and when, which it did not before. A dismissal is not merely the absence of a confirmation;
    it is the negative half of the detector's precision, and an unattributed one cannot be tied to a
    detector version or a period, so the precision it feeds would be untraceable the moment anything moved.
    """
    from datetime import UTC, datetime

    c = await db.get(ErrorCandidate, UUID(candidate_id))
    if c is None:
        return {"error": "candidate not found"}
    c.status = "dismissed"
    c.decided_by = UUID(user_id) if user_id else None
    c.decided_at = datetime.now(UTC)
    c.decision_note = note
    await db.commit()
    log.info("ed.dismissed", candidate_id=candidate_id, kind=c.kind)
    return {"candidate_id": candidate_id, "status": "dismissed"}


async def bulk_verdict(db: AsyncSession, candidate_ids: list[str], verdict: str, *,
                       user_id: str | None = None, reviewer: str = "error-detect",
                       note: str | None = None, apply_proposed: bool = True) -> dict:
    """Rule on many candidates at once.

    298,529 candidates and one verdict, with confirm and dismiss both taking a single id. At one call per
    candidate the queue is not reviewable in principle, let alone in practice, and the detectors stay
    uncalibrated forever because calibration needs verdicts in volume.

    Bulk dismissal is the common case and the reason this exists: dismissing 400 near-duplicate candidates
    is one judgement about a detector, not 400 judgements about objects. `note` records that judgement,
    which is the most useful thing in the row afterwards.

    Confirmation still goes through confirm_error per candidate rather than being reimplemented as a bulk
    UPDATE. Confirming mutates the object, writes a Review row and applies a class fix, and a second
    implementation of that would drift from the first; a wrong bulk confirm is also far more expensive than
    a wrong bulk dismiss, since it rewrites labels.
    """
    from datetime import UTC, datetime

    if verdict not in ("confirmed_error", "dismissed"):
        raise ValueError(f"unknown verdict '{verdict}'; expected confirmed_error or dismissed")
    if not candidate_ids:
        return {"verdict": verdict, "requested": 0, "applied": 0, "missing": 0, "already_decided": 0}

    ids = [UUID(c) for c in candidate_ids]
    rows = list((await db.execute(
        select(ErrorCandidate).where(ErrorCandidate.candidate_id.in_(ids)))).scalars().all())
    found = {r.candidate_id for r in rows}

    # A candidate somebody already ruled on is left alone rather than overwritten. Two reviewers working
    # the same queue is the normal case, and silently replacing the first verdict would corrupt the
    # calibration data with no trace of the disagreement.
    pending = [r for r in rows if r.status == "pending"]
    already = len(rows) - len(pending)

    applied = 0
    if verdict == "dismissed":
        stamp = datetime.now(UTC)
        for r in pending:
            r.status = "dismissed"
            r.decided_by = UUID(user_id) if user_id else None
            r.decided_at = stamp
            r.decision_note = note
            applied += 1
        await db.commit()
    else:
        for r in pending:
            res = await confirm_error(db, str(r.candidate_id), apply_proposed=apply_proposed,
                                      reviewer=reviewer, user_id=user_id)
            if "error" not in res:
                applied += 1
                r.decided_by = UUID(user_id) if user_id else None
                r.decided_at = datetime.now(UTC)
                r.decision_note = note
        await db.commit()

    out = {"verdict": verdict, "requested": len(candidate_ids), "applied": applied,
           "missing": len(ids) - len(found), "already_decided": already}
    log.info("ed.bulk_verdict", **out)
    return out


async def detector_precision(db: AsyncSession, *, confidence: float = 0.95) -> dict:
    """Per-detector precision, from the verdicts humans have actually given.

    The calibration the queue has never had. Each detector emits a score, and until now nothing said whether
    a high score meant anything: 298,529 candidates carried one verdict between them. Confirmed over
    confirmed-plus-dismissed is the detector's precision, and it is the number that decides whether a
    detector earns a place in the queue at all.

    Every rate carries a Wilson interval and a count, because with verdicts this scarce most detectors will
    report an interval spanning almost the whole range, and that is the honest answer rather than a
    discouraging one. `usable` marks the ones with enough verdicts to act on, so a detector with three
    verdicts cannot be quietly compared against one with three hundred.
    """
    from services.labelops.sampling import wilson_interval

    rows = (await db.execute(
        select(ErrorCandidate.kind, ErrorCandidate.status, func.count(ErrorCandidate.candidate_id))
        .group_by(ErrorCandidate.kind, ErrorCandidate.status))).all()

    per_kind: dict[str, dict] = {}
    for kind, status, n in rows:
        d = per_kind.setdefault(kind, {"pending": 0, "confirmed_error": 0, "dismissed": 0})
        d[status] = d.get(status, 0) + int(n)

    out = {}
    for kind, d in sorted(per_kind.items()):
        decided = d["confirmed_error"] + d["dismissed"]
        ci = wilson_interval(d["confirmed_error"], decided, confidence)
        out[kind] = {
            **d, "decided": decided, "precision": ci,
            "usable": decided >= MIN_VERDICTS_FOR_PRECISION,
            "note": None if decided >= MIN_VERDICTS_FOR_PRECISION else (
                f"only {decided} verdicts: this detector's precision is unmeasured, not low"),
        }
    total_decided = sum(v["decided"] for v in out.values())
    return {
        "per_kind": out,
        "total_candidates": sum(sum(d.values()) for d in per_kind.values()),
        "total_decided": total_decided,
        # Said plainly, because a page full of intervals from zero to one invites the reading that the
        # detectors are bad when the truth is that nobody has looked at them.
        "caveat": None if total_decided else (
            "no candidate has been ruled on, so no detector has a measurable precision. The scores rank; "
            "they do not predict."),
    }


async def summary(db: AsyncSession) -> dict:
    rows = (await db.execute(
        select(ErrorCandidate.status, func.count()).group_by(ErrorCandidate.status))).all()
    return {status: int(n) for status, n in rows}
