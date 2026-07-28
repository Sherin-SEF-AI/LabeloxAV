"""Which checks actually catch errors, measured against what humans decided.

A reasoning layer added on faith is a reasoning layer nobody can tune. Every weight in `evidence.py` is a
guess about how much a given signal is worth, and the corpus already contains the evidence to replace those
guesses with measurements: each reasoned object carries its trace in provenance, and each reviewed object
carries a human verdict.

Joining the two answers the question that matters: **when this check argued against a label, was the label
actually wrong?** That is the check's precision, and it is the only honest basis for trusting it.

The same shape as `fit_channel_reliability` in the recall path, for the same reason: a system that ranks by
hand-set numbers while the evidence to measure them accumulates unused is one that never improves.

Two things this deliberately does not do:

- **It does not auto-tune the weights.** A check's measured precision is reported and applied only when an
  operator asks, because a scoring function that silently rewrites itself from a few hundred verdicts can
  drift in a way nobody notices until a class collapses.
- **It does not count unreviewed objects.** An object nobody looked at is not evidence that the check was
  right; it is evidence that nothing was checked. Counting it would let a check earn precision by being
  applied to things nobody ever examined.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("reasoner_attribution")

# Below this many reviewed objects a check's precision is noise. Twenty-five is not a large sample either,
# but under it a single reviewer's afternoon decides whether a check is trusted.
MIN_SAMPLES = 25


async def measure_checks(db: AsyncSession, *, since_hours: int | None = None,
                         limit: int = 20000) -> dict:
    """Per check: how often it argued against a label, and how often it was right to.

    "Right" means a human subsequently rejected the object or changed its class. A check that fired on
    objects humans then accepted was wrong, however reasonable its rule looked.
    """
    from db.models import Object, Review

    stmt = (select(Object.object_id, Object.provenance, Object.state, Object.class_id)
            .where(Object.provenance.isnot(None))
            .limit(limit))
    if since_hours:
        stmt = stmt.where(Object.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
    rows = (await db.execute(stmt)).all()

    reasoned = [(oid, (prov or {}).get("reasoning"), state, cid)
                for oid, prov, state, cid in rows if (prov or {}).get("reasoning")]
    if not reasoned:
        return {"objects": 0, "reasoned": 0, "checks": {},
                "detail": "no objects carry a reasoning trace yet; run an autolabel pass with the "
                          "reasoner enabled"}

    ids = [oid for oid, _t, _s, _c in reasoned]
    # A reclassification is as much a correction as a rejection, and a check that flagged a
    # misclassification must get credit for it rather than only for outright deletions.
    reclassified = set((await db.execute(
        select(Review.object_id).where(Review.object_id.in_(ids),
                                       Review.action == "reclassify"))).scalars().all())
    reviewed = set((await db.execute(
        select(Review.object_id).where(Review.object_id.in_(ids)))).scalars().all())

    tally: dict[str, dict[str, int]] = {}
    decisions: dict[str, int] = {}
    for oid, trace, state, _cid in reasoned:
        decisions[str(trace.get("decision"))] = decisions.get(str(trace.get("decision")), 0) + 1
        if oid not in reviewed:
            # Never examined by a human, so it is evidence about nothing. Counting it would let a check
            # earn precision on objects nobody ever looked at.
            continue
        was_wrong = (str(state) == "rejected") or (oid in reclassified)
        for f in (trace.get("findings") or []):
            check = str(f.get("check"))
            weight = float(f.get("weight") or 0.0)
            t = tally.setdefault(check, {"against": 0, "against_correct": 0,
                                         "for": 0, "for_correct": 0})
            if weight < 0:
                t["against"] += 1
                t["against_correct"] += int(was_wrong)
            elif weight > 0:
                t["for"] += 1
                t["for_correct"] += int(not was_wrong)

    checks = {}
    for check, t in sorted(tally.items()):
        n_against, n_for = t["against"], t["for"]
        checks[check] = {
            "fired_against": n_against,
            "fired_for": n_for,
            # Laplace smoothed, so a check that fired three times and was right three times reads as
            # promising rather than perfect.
            "precision_against": (round((t["against_correct"] + 1) / (n_against + 2), 4)
                                  if n_against else None),
            "precision_for": (round((t["for_correct"] + 1) / (n_for + 2), 4) if n_for else None),
            "measured": bool(n_against + n_for >= MIN_SAMPLES),
            "correct_against": t["against_correct"],
            "correct_for": t["for_correct"],
        }

    reviewed_n = sum(1 for oid, _t, _s, _c in reasoned if oid in reviewed)
    log.info("reasoner.attribution", reasoned=len(reasoned), reviewed=reviewed_n,
             checks=len(checks))
    return {
        "objects": len(rows), "reasoned": len(reasoned), "reviewed": reviewed_n,
        "decisions": decisions,
        "checks": checks,
        "min_samples": MIN_SAMPLES,
        # Said out loud: precision computed over reviewed objects only is a biased sample, because review
        # is not random. It is the best available and the bias is real.
        "caveat": ("precision is measured over reviewed objects only, and review is not a random sample: "
                   "the queue deliberately surfaces uncertain objects, so these numbers describe the "
                   "check's behaviour on hard cases rather than on the corpus as a whole"),
    }


async def suggest_weights(db: AsyncSession, *, since_hours: int | None = None) -> dict:
    """What the weights would be if they followed the measurements.

    Reported, never applied. A scoring function that silently rewrites itself from a few hundred verdicts
    drifts in a way nobody notices until a class collapses, so this produces a proposal an operator reads
    and acts on rather than a change that happens.
    """
    from services.autolabel.reasoner.evidence import CHECKS

    measured = await measure_checks(db, since_hours=since_hours)
    out = {}
    for check in sorted(CHECKS):
        stats = (measured.get("checks") or {}).get(check)
        if not stats or not stats["measured"]:
            out[check] = {"status": "unmeasured",
                          "detail": f"fewer than {MIN_SAMPLES} reviewed objects carried this check"}
            continue
        p = stats["precision_against"]
        if p is None:
            out[check] = {"status": "never_fired_against"}
            continue
        # A check right 90% of the time when it objects deserves close to full weight; one right half the
        # time deserves almost none, since at 0.5 it is a coin toss dressed as evidence.
        suggested = round(max(0.0, (p - 0.5) * 2.0), 3)
        out[check] = {"status": "measured", "precision_against": p,
                      "suggested_weight_scale": suggested,
                      "detail": (f"argued against {stats['fired_against']} objects and was right "
                                 f"{stats['correct_against']} times")}
    return {"suggestions": out, "based_on": measured.get("reviewed", 0),
            "note": "reported, not applied: weights change when an operator decides they should"}


async def decision_outcomes(db: AsyncSession, *, since_hours: int | None = None) -> dict:
    """Did the reasoner's own decisions hold up?

    The headline the whole layer is accountable to: of the objects it accepted, how many did a human later
    reject? That number is the auto-accept error rate, and it is the one this exists to reduce.
    """
    from db.models import Object, Review

    stmt = select(Object.object_id, Object.provenance, Object.state)
    if since_hours:
        stmt = stmt.where(Object.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
    rows = (await db.execute(stmt.limit(20000))).all()

    traced = [(oid, (prov or {}).get("reasoning"), state)
              for oid, prov, state in rows if (prov or {}).get("reasoning")]
    if not traced:
        return {"reasoned": 0, "detail": "no reasoning traces yet"}

    ids = [oid for oid, _t, _s in traced]
    corrected = set((await db.execute(
        select(Review.object_id).where(Review.object_id.in_(ids),
                                       Review.action.in_(("reject", "reclassify"))))).scalars().all())
    reviewed = set((await db.execute(
        select(Review.object_id).where(Review.object_id.in_(ids)))).scalars().all())

    by_decision: dict[str, dict[str, int]] = {}
    for oid, trace, _state in traced:
        d = str(trace.get("decision") or "unknown")
        b = by_decision.setdefault(d, {"total": 0, "reviewed": 0, "corrected": 0})
        b["total"] += 1
        if oid in reviewed:
            b["reviewed"] += 1
            b["corrected"] += int(oid in corrected)

    out = {}
    for d, b in sorted(by_decision.items()):
        out[d] = {**b,
                  "error_rate": (round(b["corrected"] / b["reviewed"], 4) if b["reviewed"] else None)}
    return {"reasoned": len(traced), "by_decision": out,
            "headline": ("the error rate on `accept` is what this layer exists to reduce; compare it "
                         "against the same number before the reasoner was enabled")}


async def trace_for(db: AsyncSession, object_id: str) -> dict:
    """One object's reasoning, for the reviewer looking at it right now.

    This is what makes rapid review fast: the reviewer sees why the object is in front of them rather than
    having to work it out from the crop.
    """
    import uuid

    from db.models import Object

    obj = await db.get(Object, uuid.UUID(object_id))
    if obj is None:
        return {"object_id": object_id, "found": False}
    trace = (obj.provenance or {}).get("reasoning")
    return {"object_id": object_id, "found": True, "state": obj.state,
            "class_id": obj.class_id, "reasoning": trace,
            "detail": None if trace else "this object was annotated before the reasoner, or with it off"}


async def coverage(db: AsyncSession) -> dict:
    """How much of the corpus has been reasoned about at all."""
    from db.models import Object

    total = (await db.execute(select(func.count()).select_from(Object))).scalar_one()
    rows = (await db.execute(select(Object.provenance).limit(50000))).scalars().all()
    reasoned = sum(1 for p in rows if (p or {}).get("reasoning"))
    return {"objects": int(total), "sampled": len(rows), "reasoned_in_sample": reasoned,
            "fraction": round(reasoned / len(rows), 4) if rows else 0.0}
