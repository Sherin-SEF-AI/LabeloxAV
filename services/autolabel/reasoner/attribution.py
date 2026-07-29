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
# How many objects to look at when the question is 'how often does this happen across the corpus',
# which is a distribution rather than a judgement and does not need every row.
SAMPLE = 50_000


async def measure_checks(db: AsyncSession, *, since_hours: int | None = None,
                         limit: int = 20000) -> dict:
    """Per check: how often it argued against a label, and how often it was right to.

    "Right" means a human subsequently rejected the object or changed its class. A check that fired on
    objects humans then accepted was wrong, however reasonable its rule looked.
    """
    from db.models import Object, Review

    # Only reviewed objects can grade a check, so only reviewed objects are fetched. The obvious version of
    # this took an arbitrary page of the whole table and filtered afterwards, which on this corpus meant
    # 20,000 rows drawn from 583,525 to find the 2,007 that carry a human ruling: the measurement rested on
    # whichever sixty or so happened to fall in the page. `provenance` defaults to an empty object rather
    # than null, so the isnot(None) that was supposed to narrow it matched every row in the table.
    stmt = (select(Object.object_id, Object.provenance, Object.state, Object.class_id)
            .where(Object.object_id.in_(select(Review.object_id).distinct()))
            .limit(limit))
    if since_hours:
        stmt = stmt.where(Object.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
    rows = (await db.execute(stmt)).all()

    reasoned = [(oid, (prov or {}).get("reasoning"), state, cid)
                for oid, prov, state, cid in rows if (prov or {}).get("reasoning")]
    if not reasoned:
        return {"objects": len(rows), "reasoned": 0, "reviewed": 0, "checks": {},
                "detail": ("no reviewed object carries a reasoning trace yet. Either nothing has been "
                           "reasoned over, or the reasoning ran without recording traces on the objects "
                           "humans ruled on, which are the only ones a check can be graded against")}

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
    n_reviewed = n_wrong = 0
    for oid, trace, state, _cid in reasoned:
        decisions[str(trace.get("decision"))] = decisions.get(str(trace.get("decision")), 0) + 1
        if oid not in reviewed:
            # Never examined by a human, so it is evidence about nothing. Counting it would let a check
            # earn precision on objects nobody ever looked at.
            continue
        was_wrong = (str(state) == "rejected") or (oid in reclassified)
        n_reviewed += 1
        n_wrong += int(was_wrong)
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

    # What fraction of reviewed objects turned out wrong regardless of any check. Without it a precision is
    # uninterpretable: on this corpus 63% of reviewed objects were corrected, so a rule that fired at random
    # would score 0.63 and look respectable, and one scoring 0.53 looks like a coin flip when it is in fact
    # anti-correlated, firing more often on the objects that were fine. Two rules were being read that way.
    base_wrong = (n_wrong / n_reviewed) if n_reviewed else None
    base_right = (1.0 - base_wrong) if base_wrong is not None else None

    def _lift(p, base):
        if p is None or not base:
            return None
        return round(p / base, 3)

    checks = {}
    for check, t in sorted(tally.items()):
        n_against, n_for = t["against"], t["for"]
        # Laplace smoothed, so a check that fired three times and was right three times reads as promising
        # rather than perfect.
        p_against = (round((t["against_correct"] + 1) / (n_against + 2), 4) if n_against else None)
        p_for = (round((t["for_correct"] + 1) / (n_for + 2), 4) if n_for else None)
        lift_against = _lift(p_against, base_wrong)
        checks[check] = {
            "fired_against": n_against,
            "fired_for": n_for,
            "precision_against": p_against,
            "precision_for": p_for,
            # Precision over the base rate. At or below 1.0 the rule carries no information and a negative
            # weight on it actively degrades the score, which is the difference between a weak check and a
            # harmful one and is invisible from precision alone.
            "lift_against": lift_against,
            "lift_for": _lift(p_for, base_right),
            "informative": (lift_against is not None and lift_against > 1.0),
            "measured": bool(n_against + n_for >= MIN_SAMPLES),
            "correct_against": t["against_correct"],
            "correct_for": t["for_correct"],
        }

    reviewed_n = sum(1 for oid, _t, _s, _c in reasoned if oid in reviewed)
    log.info("reasoner.attribution", reasoned=len(reasoned), reviewed=reviewed_n,
             checks=len(checks))
    return {
        # Every count here is over reviewed objects, which is the population that can grade a check.
        "reviewed_objects_fetched": len(rows), "reasoned": len(reasoned), "reviewed": reviewed_n,
        "decisions": decisions,
        "checks": checks,
        "min_samples": MIN_SAMPLES,
        # The bar every check has to clear. Reported so a reader never has to guess what a precision means.
        "base_rate_wrong": (round(base_wrong, 4) if base_wrong is not None else None),
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
    # Every name that has actually been observed, not just the registered collectors. A collector may report
    # several distinct rules under their own names, and those are exactly the ones worth suggesting about:
    # a rule invisible to this list is a rule nobody can find out is bad.
    for check in sorted(set(CHECKS) | set(measured.get("checks") or {})):
        stats = (measured.get("checks") or {}).get(check)
        if not stats or not stats["measured"]:
            out[check] = {"status": "unmeasured",
                          "detail": f"fewer than {MIN_SAMPLES} reviewed objects carried this check"}
            continue
        p, lift = stats["precision_against"], stats.get("lift_against")
        if p is None:
            out[check] = {"status": "never_fired_against"}
            continue
        # Scaled by lift over the base rate, not by distance from 0.5. Using 0.5 as the neutral point was
        # wrong on this corpus, where 63% of reviewed objects were wrong anyway: it credited two rules that
        # were firing more often on the objects that turned out fine.
        suggested = round(max(0.0, min(1.0, ((lift or 0.0) - 1.0) / 0.6)), 3)
        out[check] = {"status": "measured", "precision_against": p, "lift_against": lift,
                      "informative": stats.get("informative"),
                      "suggested_weight_scale": suggested,
                      "detail": (f"argued against {stats['fired_against']} objects and was right "
                                 f"{stats['correct_against']} times, against a base rate of "
                                 f"{measured.get('base_rate_wrong')}")}
    return {"suggestions": out, "based_on": measured.get("reviewed", 0),
            "note": "reported, not applied: weights change when an operator decides they should"}


async def decision_outcomes(db: AsyncSession, *, since_hours: int | None = None) -> dict:
    """Did the reasoner's own decisions hold up?

    The headline the whole layer is accountable to: of the objects it accepted, how many did a human later
    reject? That number is the auto-accept error rate, and it is the one this exists to reduce.
    """
    from db.models import Object, Review

    # Two populations, because two different questions are being asked and one sample cannot answer both.
    # How often each decision is made is a fact about the corpus, and a broad sample answers it. Whether a
    # decision held up can only be asked of objects a human ruled on, and there the sample has to be all of
    # them: taking an arbitrary page of the whole table and filtering afterwards put the headline error rate
    # on thirty objects out of half a million, which is not a measurement, it is a coincidence.
    mix_stmt = select(Object.provenance)
    ruled_stmt = (select(Object.object_id, Object.provenance)
                  .where(Object.object_id.in_(select(Review.object_id).distinct())))
    if since_hours:
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        mix_stmt = mix_stmt.where(Object.created_at >= cutoff)
        ruled_stmt = ruled_stmt.where(Object.created_at >= cutoff)

    mix_rows = (await db.execute(mix_stmt.limit(SAMPLE))).all()
    ruled_rows = (await db.execute(ruled_stmt.limit(50000))).all()

    traced_mix = [(prov or {}).get("reasoning") for (prov,) in mix_rows if (prov or {}).get("reasoning")]
    traced_ruled = [(oid, (prov or {}).get("reasoning"))
                    for oid, prov in ruled_rows if (prov or {}).get("reasoning")]
    if not traced_mix and not traced_ruled:
        return {"reasoned": 0, "detail": "no reasoning traces yet"}

    ids = [oid for oid, _t in traced_ruled]
    corrected = set((await db.execute(
        select(Review.object_id).where(Review.object_id.in_(ids),
                                       Review.action.in_(("reject", "reclassify"))))).scalars().all())

    by_decision: dict[str, dict[str, int]] = {}
    for trace in traced_mix:
        d = str((trace or {}).get("decision") or "unknown")
        by_decision.setdefault(d, {"sampled": 0, "reviewed": 0, "corrected": 0})["sampled"] += 1
    for oid, trace in traced_ruled:
        d = str((trace or {}).get("decision") or "unknown")
        b = by_decision.setdefault(d, {"sampled": 0, "reviewed": 0, "corrected": 0})
        b["reviewed"] += 1
        b["corrected"] += int(oid in corrected)

    from services.autolabel.reasoner.verdict import PERMITS_AUTO_ACCEPT

    out = {}
    for d, b in sorted(by_decision.items()):
        rate = round(b["corrected"] / b["reviewed"], 4) if b["reviewed"] else None
        # The same count means opposite things depending on what the reasoner decided, and reporting it
        # under one name inverted three decisions out of five. A human correcting an object the reasoner
        # accepted is the reasoner having been wrong. A human correcting one it sent to review or rejected
        # is the reasoner having been right, and calling that an error rate made the layer look catastrophic
        # at exactly the thing it is best at: `reject` read as 94% wrong when it was 94% vindicated.
        if d in PERMITS_AUTO_ACCEPT:
            out[d] = {**b, "meaning": "let through without review",
                      "error_rate": rate,
                      "detail": "a human correction here is the reasoner having been wrong"}
        else:
            out[d] = {**b, "meaning": "escalated to a human",
                      "justified_rate": rate,
                      "detail": "a human correction here is the escalation having been warranted"}

    accept = out.get("accept") or {}
    return {"sampled_for_mix": len(traced_mix), "reviewed_total": len(traced_ruled),
            "by_decision": out,
            "auto_accept_error_rate": accept.get("error_rate"),
            "headline": ("the error rate on `accept` is what this layer exists to reduce; compare it "
                         "against the same number before the reasoner was enabled"),
            "caveat": ("`sampled` counts how often each decision is made across a sample of the corpus; "
                       "the per-decision rates are over every reviewed object, and review is not a random "
                       "sample: the queue deliberately surfaces uncertain objects, so these describe "
                       "behaviour on hard cases rather than on the corpus as a whole"),
            }


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
