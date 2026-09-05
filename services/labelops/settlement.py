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
2. **Sample.** Drawn randomly from the stratum's review-state objects, sized so the lot survives one
   defect: the smallest n where Wilson's upper bound on 1/n clears the tier's FAR bound (about 120 for
   far 0.05, 280 for 0.02, 565 for 0.01). Stamped with the same provenance.flywheel.cycle_id marker
   every mined batch uses, so the verdicts arrive through the existing review grid with zero new UI.
3. **Verdicts.** Read back from Review rows. An accept with the class unchanged is clean; a reject or
   ANY class edit is a defect - settlement asserts THIS class, and a within-superclass refinement is
   still a wrong label under that assertion. Unjudged samples do not enter n, but completion below 0.9
   parks the lot: skips correlate with hard crops, and a lot judged only on its easy half is not a
   measurement of the lot.
4. **Decision.** `acceptance_decision` verbatim, stored on the lot. accept -> the stratum remainder may
   be settled (chunked, revertible); reject -> the verdicts stand as reviews and the class steps down;
   inconclusive -> top up by the decision's own samples-needed figure, at most twice, then park with
   the evidence recorded.

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
MAX_TOPUPS = 2
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


async def plan_lot(db: AsyncSession, class_name: str, *, epoch: str | None = None,
                   created_by: str = "settlement") -> dict:
    """Create a lot over the class's largest (or named) stratum and stamp its sample for review.

    Refuses, with the reason, rather than creating an unmeasurable lot: population below the minimum,
    a lot already open on the stratum, or a sample the corpus cannot fill.
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

    need = sample_target(far)
    ids = (await db.execute(text(f"""
        select o.object_id from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and least(o.bbox[3] - o.bbox[1], o.bbox[4] - o.bbox[2]) >= :minside
          and {_EPOCH_SQL} = :epoch
        order by md5(o.object_id::text || :salt) limit :n"""),
        {"cid": cid, "class_name": class_name, "epoch": epoch, "minside": MIN_SIDE_PX,
         "salt": "settle-sample", "n": need})).scalars().all()
    if len(ids) < need:
        return {"error": f"the stratum holds only {len(ids)} judgeable crops (>= {MIN_SIDE_PX:.0f}px) "
                         f"of the {need} the far bound {far} needs; settle-by-sample cannot be honest "
                         "on crops nobody can judge"}

    lot_id = uuid.uuid4()
    batch_id = f"settle-{lot_id.hex[:8]}"
    await db.execute(text("""
        update object
           set provenance = coalesce(provenance, '{}'::jsonb)
                            || jsonb_build_object('flywheel',
                                 coalesce(provenance->'flywheel', '{}'::jsonb)
                                 || jsonb_build_object('cycle_id', cast(:bid as text),
                                                       'reason', 'settlement acceptance sample'))
         where object_id = any(cast(:ids as uuid[]))"""),
        {"bid": batch_id, "ids": [str(i) for i in ids]})

    db.add(SettlementLot(lot_id=lot_id, class_id=cid, model_epoch=epoch, population=population,
                         tier=tier, far_bound=far, sample_object_ids=[str(i) for i in ids],
                         batch_id=batch_id, status="judging", created_by=created_by))
    await db.commit()
    log.info("settlement.lot_planned", lot=str(lot_id), cls=class_name, epoch=epoch,
             population=population, sample=len(ids), far=far, tier=tier)
    return {"lot_id": str(lot_id), "class_name": class_name, "epoch": epoch,
            "population": population, "tier": tier, "far_bound": far, "sample_n": len(ids),
            "batch_id": batch_id,
            "review_at": f"/review/grid?flywheel={batch_id}&states=review",
            "human_minutes_estimate": round(len(ids) / 10)}


async def tally_lot(db: AsyncSession, lot_id: str) -> dict:
    """Read the sample's human verdicts and run the acceptance decision.

    A clean verdict is an accept/confirm that left the class alone. A reject or ANY class edit is a
    defect: the lot asserts THIS class, so a refinement inside the superclass is still a wrong label
    under the assertion being tested. Unjudged crops stay out of n; below the completion floor the lot
    parks instead of deciding, because skips correlate with hard crops.
    """
    from services.labelops.sampling import acceptance_decision

    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status not in ("judging",):
        return {"lot_id": str(lot.lot_id), "status": lot.status,
                "detail": "only a judging lot tallies"}

    sample_ids = [uuid.UUID(s) for s in (lot.sample_object_ids or [])]
    rows = (await db.execute(
        select(Review.object_id, Review.action, Review.before, Review.after, Review.ts_ns)
        .where(Review.object_id.in_(sample_ids))
        .order_by(Review.ts_ns))).all()

    latest: dict[uuid.UUID, tuple] = {r[0]: r for r in rows}   # last ruling per object stands
    n = defects = 0
    for _oid, action, before, after, _ts in latest.values():
        if action not in ("accept", "confirm", "reject", "reclassify"):
            continue
        n += 1
        changed_class = (before or {}).get("class_id") != (after or {}).get("class_id")
        if action == "reject" or changed_class:
            defects += 1

    skips = len(sample_ids) - n
    completion = n / len(sample_ids) if sample_ids else 0.0
    lot.sample_n, lot.defects, lot.skips = n, defects, skips

    if completion < COMPLETION_FLOOR:
        await db.commit()
        return {"lot_id": str(lot.lot_id), "status": lot.status, "sample_n": n, "defects": defects,
                "skips": skips, "completion": round(completion, 3),
                "detail": f"waiting: {skips} of {len(sample_ids)} crops unjudged; a lot judged only "
                          f"on its willing half is not a measurement (floor {COMPLETION_FLOOR})"}

    decision = acceptance_decision(defects, n, max_defect_rate=lot.far_bound)
    lot.decision = decision

    if decision["verdict"] == "accept":
        lot.status, lot.decided_at = "accepted", datetime.now(UTC)
    elif decision["verdict"] == "reject":
        lot.status, lot.decided_at = "rejected", datetime.now(UTC)
    elif lot.topups >= MAX_TOPUPS:
        lot.status, lot.decided_at = "parked", datetime.now(UTC)
        decision = {**decision,
                    "parked": f"inconclusive after {MAX_TOPUPS} top-ups; the evidence is recorded and "
                              "a person decides"}
        lot.decision = decision
    await db.commit()
    log.info("settlement.tallied", lot=str(lot.lot_id), verdict=decision["verdict"],
             n=n, defects=defects, status=lot.status)
    return {"lot_id": str(lot.lot_id), "status": lot.status, "sample_n": n, "defects": defects,
            "skips": skips, "completion": round(completion, 3), "decision": decision}


async def top_up_lot(db: AsyncSession, lot_id: str) -> dict:
    """Extend an inconclusive lot's sample by its own samples-needed figure. At most MAX_TOPUPS."""
    lot = await db.get(SettlementLot, uuid.UUID(str(lot_id)))
    if lot is None:
        return {"error": "lot not found"}
    if lot.status != "judging" or (lot.decision or {}).get("verdict") != "inconclusive":
        return {"error": f"only a judging, inconclusive lot tops up (status={lot.status})"}
    if lot.topups >= MAX_TOPUPS:
        return {"error": f"already topped up {MAX_TOPUPS} times; the lot parks instead of sampling "
                         "forever"}

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    class_name = onto.by_id(lot.class_id).name
    have = set(lot.sample_object_ids or [])
    extra = max(25, sample_target(lot.far_bound) // 2)
    ids = (await db.execute(text(f"""
        select o.object_id from object o
        where o.class_id = :cid and o.state = 'review' and o.source <> 'human'
          and least(o.bbox[3] - o.bbox[1], o.bbox[4] - o.bbox[2]) >= :minside
          and {_EPOCH_SQL} = :epoch
          and not (o.object_id::text = any(cast(:have as text[])))
        order by md5(o.object_id::text || :salt) limit :n"""),
        {"cid": lot.class_id, "class_name": class_name, "epoch": lot.model_epoch,
         "minside": MIN_SIDE_PX, "have": list(have), "salt": f"settle-topup-{lot.topups}",
         "n": extra})).scalars().all()
    if not ids:
        return {"error": "the stratum has no more judgeable crops to draw"}

    await db.execute(text("""
        update object
           set provenance = coalesce(provenance, '{}'::jsonb)
                            || jsonb_build_object('flywheel',
                                 coalesce(provenance->'flywheel', '{}'::jsonb)
                                 || jsonb_build_object('cycle_id', cast(:bid as text),
                                                       'reason', 'settlement sample top-up'))
         where object_id = any(cast(:ids as uuid[]))"""),
        {"bid": lot.batch_id, "ids": [str(i) for i in ids]})
    lot.sample_object_ids = [*have, *(str(i) for i in ids)]
    lot.topups += 1
    await db.commit()
    return {"lot_id": str(lot.lot_id), "added": len(ids), "topups": lot.topups,
            "sample_total": len(lot.sample_object_ids)}


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
            "spots": len(spot_ids), "status": "settled"}


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
