"""Acceptance-sampled settlement: the machine may close a label only under a passed lot.

90.5% of this corpus sits in `state='review'` forever because the only paths out were a person per
object or nothing. The operator's decision (2026-09-01): the machine may settle labels, per class,
evidence-gated, sampled-QA'd, revertible, with automatic step-down. This module is the evidence part -
the first real caller of `sampling.py::acceptance_decision`, which shipped as a textbook lot-acceptance
primitive with zero callers.

The shape of a lot, end to end:

1. **Stratum.** (class_id, model_epoch), where the epoch is the model_version of the winning proposal
   in provenance (pre-attribution objects pool as one frozen "legacy" epoch). One detector asserting
   one class is one population with one defect rate; mixing epochs would let a good new model launder
   a bad old one's labels. Minimum population 2,000 - below that, reviewing the queue is cheaper than
   an acceptance sample.
2. **Sample.** Drawn randomly from the stratum's review-state objects as a prefix of ONE deterministic
   permutation (md5 over object_id with a fixed salt), in increments of `SPRT_INCREMENT`, up to a cap
   sized so a fixed draw would survive one defect: the smallest n where Wilson's upper bound on 1/n
   clears the tier's FAR bound (about 120 for far 0.05, 280 for 0.02, 565 for 0.01). The sample at the
   cap is therefore identical to the fixed draw it replaced. Stamped with the same
   provenance.flywheel.cycle_id marker every mined batch uses, so the verdicts arrive through the
   existing review grid with zero new UI; the grid serves a settlement batch in draw order, because a
   hardest-first prefix is not a random sample of the lot.
3. **Verdicts.** Read back from Review rows. An accept with the class unchanged is clean; a reject or
   ANY class edit is a defect - settlement asserts THIS class, and a within-superclass refinement is
   still a wrong label under that assertion. Unjudged samples do not enter n, but completion below 0.9
   parks the lot: skips correlate with hard crops, and a lot judged only on its easy half is not a
   measurement of the lot.
4. **Decision.** Sequential (Wald's SPRT, `sampling.py::sprt_decision`, bad rate = the far bound,
   good rate = half of it) over the completed increments only, so a half-judged increment never
   decides. Accept is conjunctive: the SPRT must cross its accept bound AND `acceptance_decision`
   (Wilson, the fixed-sample rule) must accept on the same verdicts, so nothing settles on weaker
   evidence than before; the gain is that a bad class rejects after a handful of defects and a clean
   one stops asking early. "Continue" draws the next increment; at the cap the Wilson rule alone
   decides exactly as it always did, with the legacy top-up (at most twice) then park. accept -> the
   stratum remainder may be settled (chunked, revertible); reject -> the verdicts stand as reviews
   and the class steps down. Lots planned before the sequential rule (`rule='wilson'`) keep deciding
   by Wilson alone.

Settling writes `state='settled'` and nothing else: source keeps naming the machine that proposed the
label, `accepted` keeps meaning "a person ruled", and every write carries an AgentRun id so one revert
returns the lot to `review`.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object, Review, SettlementLot

log = get_logger("labelops.settlement")

MIN_POPULATION = 2_000
MAX_TOPUPS = 2           # post-cap Wilson top-ups; the sequential increments below the cap are unbounded
SPRT_INCREMENT = 25      # verdicts drawn per sequential step; one increment is one grid sitting
SPRT_ALPHA = 0.05        # P(reject | good rate)
SPRT_BETA = 0.10         # P(accept | bad rate)
SAMPLE_SALT = "settle-sample"   # one permutation per stratum; every increment is a prefix of it
VERDICTS_PER_MINUTE = 10        # the grid's measured pace, shared with human_minutes_estimate
COMPLETION_FLOOR = 0.9
CHUNK = 5_000            # objects per settlement AgentRun (the 137k-object revert lesson)
SPOT_FRACTION = 0.02
SPOT_CAP = 200
MIN_SIDE_PX = 12.0       # the same judgeability floor the precision batches use

# The winning proposal's model, straight from provenance: the first proposal that was not overruled and
# asserts the object's current class. Shared by the population count and the sample draw, because a lot
# whose population and sample were defined by different predicates is not a lot.
_EPOCH_SQL = ("coalesce(jsonb_path_query_first(o.provenance, "
              "'$.proposals[*] ? (@.verdict != \"overruled\") ? (@.class_name == $cn).model_version', "
              "jsonb_build_object('cn', cast(:class_name as text))) #>> '{}', 'legacy')")


def tier_for(class_name: str) -> tuple[str, float]:
    """(tier, FAR bound) for a class, from the active pack's safety policy."""
    from packs.registry import default_pack_id, get_pack

    policy = get_pack(default_pack_id()).safety_policy
    if class_name in policy.critical_class_names():
        tier = "critical"
    elif policy.is_safety_class(class_name):
        tier = "safety"
    else:
        tier = "default"
    return tier, float(policy.accept_far_bound(class_name))


def sample_target(far_bound: float, *, survivable_defects: int = 1) -> int:
    """The smallest n whose Wilson upper bound on `survivable_defects` defects clears the FAR bound.

    Stated up front so the human cost is known before anybody is asked: about 120 clean-ish verdicts
    for far 0.05, 280 for 0.02, 565 for 0.01. Planning for zero defects would park every lot the first
    time a reviewer finds one real mistake in a population allowed to contain some.
    """
    from services.labelops.sampling import wilson_interval

    n = survivable_defects + 1
    while n < 100_000:
        if wilson_interval(survivable_defects, n)["hi"] <= far_bound:
            return n
        n = n + max(1, n // 50)
    raise ValueError(f"no practical sample size clears far bound {far_bound}")


async def stratum_population(db: AsyncSession, class_id: int, class_name: str, epoch: str) -> int:
    return int((await db.execute(text(f"""
        select count(*) from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and {_EPOCH_SQL} = :epoch"""), {"cid": class_id, "class_name": class_name,
                                          "epoch": epoch})).scalar_one())


async def strata_for_class(db: AsyncSession, class_id: int, class_name: str) -> list[dict]:
    """Every epoch this class's review-state objects froze under, largest first."""
    rows = (await db.execute(text(f"""
        select {_EPOCH_SQL} epoch, count(*) n from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
        group by 1 order by 2 desc"""), {"cid": class_id, "class_name": class_name})).all()
    return [{"epoch": r[0], "population": int(r[1])} for r in rows]


async def _draw(db: AsyncSession, *, class_id: int, class_name: str, epoch: str,
                exclude: list[str], n: int) -> list[uuid.UUID]:
    """The next `n` judgeable crops of the stratum in the permutation, after those already drawn."""
    if n <= 0:
        return []
    return list((await db.execute(text(f"""
        select o.object_id from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and least(o.bbox[3] - o.bbox[1], o.bbox[4] - o.bbox[2]) >= :minside
          and {_EPOCH_SQL} = :epoch
          and not (o.object_id::text = any(cast(:have as text[])))
        order by md5(o.object_id::text || :salt) limit :n"""),
        {"cid": class_id, "class_name": class_name, "epoch": epoch, "minside": MIN_SIDE_PX,
         "have": list(exclude), "salt": SAMPLE_SALT, "n": n})).scalars().all())


async def _judgeable(db: AsyncSession, *, class_id: int, class_name: str, epoch: str) -> int:
    return int((await db.execute(text(f"""
        select count(*) from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and least(o.bbox[3] - o.bbox[1], o.bbox[4] - o.bbox[2]) >= :minside
          and {_EPOCH_SQL} = :epoch"""),
        {"cid": class_id, "class_name": class_name, "epoch": epoch,
         "minside": MIN_SIDE_PX})).scalar_one())


async def _stamp(db: AsyncSession, ids: list, *, batch_id: str, reason: str) -> None:
    await db.execute(text("""
        update object
           set provenance = coalesce(provenance, '{}'::jsonb)
                            || jsonb_build_object('flywheel',
                                 coalesce(provenance->'flywheel', '{}'::jsonb)
                                 || jsonb_build_object('cycle_id', cast(:bid as text),
                                                       'reason', cast(:reason as text)))
         where object_id = any(cast(:ids as uuid[]))"""),
        {"bid": batch_id, "reason": reason, "ids": [str(i) for i in ids]})


def sprt_params(far_bound: float) -> dict:
    """The sequential test's fixed parameters for a tier: bad rate = the far bound, good rate = half."""
    from services.labelops.sampling import sprt_bounds

    b = sprt_bounds(alpha=SPRT_ALPHA, beta=SPRT_BETA)
    return {"p0": far_bound, "p1": far_bound / 2, "alpha": SPRT_ALPHA, "beta": SPRT_BETA,
            "bound_accept": round(b["bound_accept"], 4), "bound_reject": round(b["bound_reject"], 4),
            "increment": SPRT_INCREMENT}


async def plan_lot(db: AsyncSession, class_name: str, *, epoch: str | None = None,
                   created_by: str = "settlement") -> dict:
    """Create a lot over the class's largest (or named) stratum and stamp its first increment.

    Refuses, with the reason, rather than creating an unmeasurable lot: population below the minimum,
    a lot already open on the stratum, or fewer judgeable crops than the cap the far bound needs (the
    sequential test may run to the cap, and a lot that cannot be completed is not started).
    """
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if not onto.has_name(class_name):
        return {"error": f"'{class_name}' is not in the ontology"}
    cid = onto.by_name(class_name).id
    tier, far = tier_for(class_name)

    if epoch is None:
        strata = await strata_for_class(db, cid, class_name)
        if not strata:
            return {"error": f"no review-state {class_name} objects to settle"}
        epoch = strata[0]["epoch"]
        population = strata[0]["population"]
    else:
        population = await stratum_population(db, cid, class_name, epoch)

    if population < MIN_POPULATION:
        return {"error": f"stratum ({class_name}, {epoch}) holds {population} objects; below "
                         f"{MIN_POPULATION} an acceptance sample costs more than reviewing the queue"}

    open_lot = (await db.execute(select(SettlementLot).where(
        SettlementLot.class_id == cid, SettlementLot.model_epoch == epoch,
        SettlementLot.status.in_(("sampling", "judging", "accepted"))))).scalars().first()
    if open_lot is not None:
        return {"error": f"lot {open_lot.lot_id} is already {open_lot.status} on this stratum"}

    cap_n = sample_target(far)
    judgeable = await _judgeable(db, class_id=cid, class_name=class_name, epoch=epoch)
    if judgeable < cap_n:
        return {"error": f"the stratum holds only {judgeable} judgeable crops (>= {MIN_SIDE_PX:.0f}px) "
                         f"of the {cap_n} the far bound {far} needs; settle-by-sample cannot be honest "
                         "on crops nobody can judge"}
    first = min(SPRT_INCREMENT, cap_n)
    ids = await _draw(db, class_id=cid, class_name=class_name, epoch=epoch, exclude=[], n=first)
    if len(ids) < first:
        return {"error": f"drew {len(ids)} of the first {first} crops; the stratum changed under the plan"}

    lot_id = uuid.uuid4()
    batch_id = f"settle-{lot_id.hex[:8]}"
    await _stamp(db, ids, batch_id=batch_id, reason="settlement acceptance sample")

    now = datetime.now(UTC)
    db.add(SettlementLot(lot_id=lot_id, class_id=cid, model_epoch=epoch, population=population,
                         tier=tier, far_bound=far, sample_object_ids=[str(i) for i in ids],
                         batch_id=batch_id, status="judging", created_by=created_by,
                         rule="sprt", cap_n=cap_n, sprt={**sprt_params(far), "trajectory": []},
                         increments=[{"at": now.isoformat(), "added": len(ids),
                                      "n_after": len(ids)}]))
    await db.commit()
    log.info("settlement.lot_planned", lot=str(lot_id), cls=class_name, epoch=epoch,
             population=population, sample=len(ids), cap_n=cap_n, far=far, tier=tier)
    return {"lot_id": str(lot_id), "class_name": class_name, "epoch": epoch,
            "population": population, "tier": tier, "far_bound": far, "sample_n": len(ids),
            "cap_n": cap_n, "rule": "sprt", "batch_id": batch_id,
            "review_at": f"/review/grid?flywheel={batch_id}&states=review",
            "human_minutes_estimate": round(len(ids) / VERDICTS_PER_MINUTE)}


async def _verdicts(db: AsyncSession, sample_ids: list[uuid.UUID]) -> dict[uuid.UUID, bool]:
    """The latest human ruling per sampled object: True for clean, False for a defect. Unjudged
    objects are absent. A reject or ANY class edit is a defect under the class the lot asserts."""
    if not sample_ids:
        return {}
    rows = (await db.execute(
        select(Review.object_id, Review.action, Review.before, Review.after, Review.ts_ns)
        .where(Review.object_id.in_(sample_ids))
        .order_by(Review.ts_ns))).all()
    latest: dict[uuid.UUID, tuple] = {r[0]: r for r in rows}   # last ruling per object stands
    out: dict[uuid.UUID, bool] = {}
    for oid, action, before, after, _ts in latest.values():
        if action not in ("accept", "confirm", "reject", "reclassify"):
            continue
        changed_class = (before or {}).get("class_id") != (after or {}).get("class_id")
        out[oid] = not (action == "reject" or changed_class)
    return out


def _completed_prefix(lot: SettlementLot, judged: dict[uuid.UUID, bool]) -> tuple[list[str], int]:
    """The sampled ids covered by the leading run of completed increments, and how many increments
    that is. An increment is complete at `COMPLETION_FLOOR` of its crops judged. A lot without
    increment records (planned before the sequential rule) is one increment."""
    ids = list(lot.sample_object_ids or [])
    incs = list(lot.increments or []) or [{"added": len(ids)}]
    prefix: list[str] = []
    done = 0
    pos = 0
    for inc in incs:
        chunk = ids[pos:pos + int(inc.get("added", 0))]
        pos += len(chunk)
        if not chunk:
            break
        judged_here = sum(1 for s in chunk if uuid.UUID(s) in judged)
        if judged_here / len(chunk) < COMPLETION_FLOOR:
            break
        prefix.extend(chunk)
        done += 1
    return prefix, done


async def tally_lot(db: AsyncSession, lot_id: str) -> dict:
    """Read the sample's human verdicts and run the acceptance decision.

    A clean verdict is an accept/confirm that left the class alone. A reject or ANY class edit is a
    defect: the lot asserts THIS class, so a refinement inside the superclass is still a wrong label
    under the assertion being tested. Unjudged crops stay out of n; an increment below the completion
    floor does not enter the tally at all, because skips correlate with hard crops and the verdicts
    arrive in an order.

    Sequential lots decide by `sprt_decision` over the completed prefix, accept conjunctive with the
    Wilson rule, reject on the SPRT alone; "continue" surfaces as `inconclusive` so the agent's
    existing top-up hook draws the next increment. At the cap the Wilson rule decides verbatim.
    """
    from services.labelops.sampling import acceptance_decision, sprt_decision

    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status not in ("judging",):
        return {"lot_id": str(lot.lot_id), "status": lot.status,
                "detail": "only a judging lot tallies"}

    sample_ids = [uuid.UUID(s) for s in (lot.sample_object_ids or [])]
    judged = await _verdicts(db, sample_ids)
    sequential = lot.rule == "sprt" and (lot.cap_n or 0) > 0
    drawn = len(sample_ids)
    at_cap = drawn >= (lot.cap_n or 0)

    if sequential and not at_cap:
        prefix, done = _completed_prefix(lot, judged)
        counted = [uuid.UUID(s) for s in prefix]
    else:
        # Fixed-sample rule over the whole draw; the completion floor applies to all of it.
        counted = [oid for oid in sample_ids if oid in judged]
        done = len(lot.increments or []) or 1
        prefix = [str(o) for o in sample_ids] if len(counted) / max(1, drawn) >= COMPLETION_FLOOR else []

    n = sum(1 for oid in counted if oid in judged)
    defects = sum(1 for oid in counted if oid in judged and not judged[oid])
    judged_total = len(judged)
    skips = drawn - judged_total
    completion = judged_total / drawn if drawn else 0.0
    lot.sample_n, lot.defects, lot.skips = n, defects, skips

    if not prefix:
        await db.commit()
        pending = drawn - len(judged)
        return {"lot_id": str(lot.lot_id), "status": lot.status, "sample_n": n, "defects": defects,
                "skips": skips, "completion": round(completion, 3),
                "detail": f"waiting: {pending} of {drawn} crops unjudged; a lot judged only "
                          f"on its willing half is not a measurement (floor {COMPLETION_FLOOR})"}

    wilson = acceptance_decision(defects, n, max_defect_rate=lot.far_bound)
    now = datetime.now(UTC)
    if sequential and not at_cap:
        params = lot.sprt or sprt_params(lot.far_bound)
        sp = sprt_decision(defects, n, p0=params["p0"], p1=params["p1"],
                           alpha=params.get("alpha", SPRT_ALPHA), beta=params.get("beta", SPRT_BETA))
        traj = list(params.get("trajectory") or [])
        if not traj or (traj[-1]["n"], traj[-1]["defects"]) != (n, defects):
            traj.append({"n": n, "defects": defects, "llr": sp["llr"], "at": now.isoformat()})
        lot.sprt = {**params, "trajectory": traj, "expected_remaining": sp["expected_remaining"],
                    "oc": sp["oc"], "p_hat": sp["p_hat"]}
        lot.llr = sp["llr"]
        if sp["verdict"] == "accept" and wilson["verdict"] == "accept":
            verdict, reason = "accept", f"sequential: {sp['reason']}; fixed-sample: {wilson['reason']}"
        elif sp["verdict"] == "reject":
            verdict, reason = "reject", f"sequential: {sp['reason']}"
        elif sp["verdict"] == "accept":
            verdict = "inconclusive"
            reason = (f"sequential accept ({sp['reason']}) but the fixed-sample rule does not agree "
                      f"({wilson['reason']}); another increment")
        else:
            verdict, reason = "inconclusive", f"{sp['reason']}; another increment"
        decision = {"verdict": verdict, "reason": reason, "rule": "sprt", "n": n, "defects": defects,
                    "increments_complete": done, "sprt": sp, "wilson": wilson,
                    "next_increment": min(SPRT_INCREMENT, (lot.cap_n or 0) - drawn)}
    else:
        decision = {**wilson, "rule": "wilson" if not sequential else "wilson_at_cap",
                    "increments_complete": done, "wilson": wilson}
        if sequential:
            params = lot.sprt or sprt_params(lot.far_bound)
            sp = sprt_decision(defects, n, p0=params["p0"], p1=params["p1"],
                               alpha=params.get("alpha", SPRT_ALPHA),
                               beta=params.get("beta", SPRT_BETA))
            decision["sprt"] = sp
            lot.llr = sp["llr"]
            decision["reason"] = f"at the cap of {lot.cap_n}: fixed-sample rule decides; {wilson['reason']}"
    lot.decision = decision

    if decision["verdict"] == "accept":
        lot.status, lot.decided_at = "accepted", now
    elif decision["verdict"] == "reject":
        lot.status, lot.decided_at = "rejected", now
    elif (not sequential or at_cap) and lot.topups >= MAX_TOPUPS:
        lot.status, lot.decided_at = "parked", now
        decision = {**decision,
                    "parked": f"inconclusive after {MAX_TOPUPS} top-ups; the evidence is recorded and "
                              "a person decides"}
        lot.decision = decision
    await db.commit()
    log.info("settlement.tallied", lot=str(lot.lot_id), verdict=decision["verdict"],
             n=n, defects=defects, status=lot.status, rule=decision.get("rule"))
    return {"lot_id": str(lot.lot_id), "status": lot.status, "sample_n": n, "defects": defects,
            "skips": skips, "completion": round(completion, 3), "decision": decision}


async def top_up_lot(db: AsyncSession, lot_id: str) -> dict:
    """Extend an undecided lot's sample.

    Below the cap a sequential lot draws its next increment (the next `SPRT_INCREMENT` of the same
    permutation, unbounded: the cap is the stop). At or past the cap, and for lots planned under the
    fixed rule, the legacy top-up applies: half the fixed sample, at most MAX_TOPUPS times.
    """
    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status != "judging" or (lot.decision or {}).get("verdict") != "inconclusive":
        return {"error": f"only a judging, inconclusive lot tops up (status={lot.status})"}

    have = list(lot.sample_object_ids or [])
    sequential = lot.rule == "sprt" and (lot.cap_n or 0) > 0
    below_cap = sequential and len(have) < lot.cap_n
    if below_cap:
        # The next increment waits for the current one: drawing ahead of the judging would put two
        # half-judged increments on the grid and neither would ever tally.
        done = int((lot.decision or {}).get("increments_complete") or 0)
        if done < len(lot.increments or []):
            return {"error": f"increment {len(lot.increments)} is still being judged; the next one "
                             "is drawn when it completes"}
        extra = min(SPRT_INCREMENT, lot.cap_n - len(have))
    else:
        if lot.topups >= MAX_TOPUPS:
            return {"error": f"already topped up {MAX_TOPUPS} times; the lot parks instead of sampling "
                             "forever"}
        extra = max(25, sample_target(lot.far_bound) // 2)

    from services.autolabel.ontology import get_ontology

    class_name = get_ontology().by_id(lot.class_id).name
    ids = await _draw(db, class_id=lot.class_id, class_name=class_name, epoch=lot.model_epoch,
                      exclude=have, n=extra)
    if not ids:
        return {"error": "the stratum has no more judgeable crops to draw"}

    await _stamp(db, ids, batch_id=lot.batch_id,
                 reason="settlement sample increment" if below_cap else "settlement sample top-up")
    lot.sample_object_ids = [*have, *(str(i) for i in ids)]
    lot.increments = [*(lot.increments or []),
                      {"at": datetime.now(UTC).isoformat(), "added": len(ids),
                       "n_after": len(lot.sample_object_ids),
                       "kind": "increment" if below_cap else "topup"}]
    if not below_cap:
        lot.topups += 1
    await db.commit()
    return {"lot_id": str(lot.lot_id), "added": len(ids), "topups": lot.topups,
            "increments": len(lot.increments), "sample_total": len(lot.sample_object_ids),
            "cap_n": lot.cap_n}


def expected_remaining(lot: SettlementLot) -> dict:
    """What this lot still costs and what it is worth: the verdict-minute allocator's one number.

    `remaining` is Wald's ASN from the lot's current position (fixed-rule lots: the crops still
    unjudged), clamped to what the cap allows; `minutes` at the grid's pace; `value` is the objects a
    passed lot would settle, weighted by the probability it passes at the smoothed observed rate,
    per minute of a person's time. A lot that has decided is worth nothing more to judge.
    """
    from services.labelops.sampling import sprt_expected_remaining, sprt_oc, sprt_p_hat

    drawn = len(lot.sample_object_ids or [])
    n, defects = int(lot.sample_n or 0), int(lot.defects or 0)
    if lot.status != "judging":
        return {"remaining": 0, "minutes": 0.0, "value": 0.0, "oc": None}
    sequential = lot.rule == "sprt" and (lot.cap_n or 0) > 0
    p0, p1 = lot.far_bound, lot.far_bound / 2
    p_hat = sprt_p_hat(defects, n, p0=p0, p1=p1)
    oc = sprt_oc(p_hat, p0=p0, p1=p1, alpha=SPRT_ALPHA, beta=SPRT_BETA)
    if sequential:
        asn = sprt_expected_remaining(defects, n, p0=p0, p1=p1, alpha=SPRT_ALPHA, beta=SPRT_BETA)
        unjudged = max(0, drawn - n)
        # At least the crops already drawn and unjudged; at most what the cap still allows.
        remaining = int(round(min(max(asn, unjudged), max(0, lot.cap_n - n))))
        if remaining == 0 and unjudged == 0 and (lot.decision or {}).get("verdict") == "inconclusive":
            remaining = min(SPRT_INCREMENT, max(0, lot.cap_n - drawn))
    else:
        remaining = max(0, drawn - n)
    minutes = remaining / VERDICTS_PER_MINUTE
    value = (lot.population * oc / minutes) if minutes > 0 else 0.0
    return {"remaining": remaining, "minutes": round(minutes, 1), "value": round(value, 1),
            "oc": round(oc, 4)}


async def settle_lot(db: AsyncSession, lot_id: str, *, created_by: str = "settlement") -> dict:
    """Write 'settled' onto the stratum remainder of an accepted lot, chunked and revertible.

    Every guard re-checked here rather than trusted from the caller: the operator switch, the kill
    switch, the tier rule (critical never auto-applies - this function refuses it outright; a person
    settles a critical lot through their own explicit endpoint if that day ever comes), and the lot's
    own status. Each chunk is one AgentRun whose changes restore state and source, so the whole lot
    reverts run by run.
    """
    from services.govern.killswitch import get_state

    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status != "accepted":
        return {"error": f"only an accepted lot settles; this one is {lot.status}"}
    if lot.tier == "critical":
        return {"error": "a critical-tier lot never auto-applies; the 0.01 bound means a person "
                         "makes the final write"}
    state = await get_state(db)
    if not state.loop_enabled:
        return {"error": "the kill switch is engaged; nothing settles"}
    if not state.settlement_enabled:
        return {"error": "settlement_enabled is off; the lot stays accepted and waits for the "
                         "operator switch"}

    from services.autolabel.ontology import get_ontology

    class_name = get_ontology().by_id(lot.class_id).name
    sample = set(lot.sample_object_ids or [])

    # Session by session, so each chunk's blast radius is one drive and the run list reads as a map.
    session_rows = (await db.execute(text(f"""
        select f.session_id, array_agg(o.object_id) ids
        from object o join frame f on f.frame_id = o.frame_id
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and {_EPOCH_SQL} = :epoch
        group by f.session_id order by f.session_id"""),
        {"cid": lot.class_id, "class_name": class_name, "epoch": lot.model_epoch})).all()

    run_ids: list[str] = list(lot.run_ids or [])
    settled_total = 0
    spot_ids: list[uuid.UUID] = []
    spot_batch = f"spot-{lot.lot_id.hex[:8]}"
    for session_id, ids in session_rows:
        ids = [i for i in ids if str(i) not in sample]
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            run_id = uuid.uuid4()
            changes: dict[str, dict] = {}
            objs = (await db.execute(select(Object).where(Object.object_id.in_(chunk)))).scalars().all()
            for obj in objs:
                if obj.state != "review" or obj.source == "human":
                    continue
                changes[str(obj.object_id)] = {"from_state": obj.state, "from_source": obj.source}
                obj.state = "settled"
                obj.provenance = {**(obj.provenance or {}),
                                  "agent_run_id": str(run_id),
                                  "settlement": {"lot_id": str(lot.lot_id),
                                                 "epoch": lot.model_epoch,
                                                 "far_bound": lot.far_bound}}
                obj.version = (obj.version or 0) + 1
                # The spot mirror: a deterministic 2% by a different hash salt than the acceptance
                # sample, so the continuous check never re-examines the crops the decision was made on.
                digest = hashlib.md5(f"{obj.object_id}settle-spot".encode()).hexdigest()
                if int(digest[:8], 16) / 0xFFFFFFFF < SPOT_FRACTION and len(spot_ids) < SPOT_CAP:
                    spot_ids.append(obj.object_id)
                    # Stamped like a mined batch so the grid can serve it; without this the spots
                    # sat on settled objects no default triage view ever listed, and no spot was
                    # ever judged.
                    obj.provenance = {**obj.provenance,
                                      "flywheel": {**(obj.provenance.get("flywheel") or {}),
                                                   "cycle_id": spot_batch,
                                                   "reason": "settlement spot check"}}
            db.add(AgentRun(run_id=run_id, kind="settlement",
                            scope={"lot_id": str(lot.lot_id), "session_id": str(session_id)},
                            status="committed", policy={"far_bound": lot.far_bound,
                                                        "epoch": lot.model_epoch},
                            counts={"settled": len(changes)}, changes=changes, critic={},
                            created_by=created_by))
            run_ids.append(str(run_id))
            settled_total += len(changes)
            lot.run_ids = run_ids
            await db.commit()   # chunk by chunk: an interruption keeps everything already settled

    from db.models import SettlementSpot

    for oid in spot_ids:
        db.add(SettlementSpot(lot_id=lot.lot_id, object_id=oid))
    lot.spot_total = len(spot_ids)
    lot.status = "settled"
    await db.commit()

    from services.govern.audit import record

    await record(db, "settlement", "settle", str(lot.lot_id),
                 {"class_name": class_name, "epoch": lot.model_epoch, "settled": settled_total,
                  "runs": len(run_ids), "spots": len(spot_ids), "decision": lot.decision})
    log.info("settlement.settled", lot=str(lot.lot_id), settled=settled_total, runs=len(run_ids))
    return {"lot_id": str(lot.lot_id), "settled": settled_total, "runs": run_ids,
            "spots": len(spot_ids), "status": "settled", "spot_batch": spot_batch,
            "spot_review_at": f"/review/grid?flywheel={spot_batch}&states=settled"}


async def revert_lot(db: AsyncSession, lot_id: str, *, reason: str) -> dict:
    """Return every object the lot settled to 'review' - the conservative direction, the only
    automatic one - by reverting the lot's own runs."""
    from services.agent.runs import revert_run

    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status != "settled":
        return {"error": f"only a settled lot reverts; this one is {lot.status}"}

    reverted = skipped = 0
    for rid in lot.run_ids or []:
        try:
            r = await revert_run(db, uuid.UUID(rid))
            reverted += r.get("reverted", 0)
            skipped += r.get("skipped", 0)
        except ValueError as exc:
            log.warning("settlement.revert_run_skipped", run=rid, error=str(exc))
    lot.status = "reverted"
    await db.commit()

    from services.govern.audit import record

    await record(db, "settlement", "revert", str(lot.lot_id),
                 {"reason": reason, "reverted": reverted, "skipped": skipped})
    log.info("settlement.reverted", lot=str(lot.lot_id), reverted=reverted, skipped=skipped)
    return {"lot_id": str(lot.lot_id), "status": "reverted", "reverted": reverted,
            "skipped": skipped, "reason": reason}
