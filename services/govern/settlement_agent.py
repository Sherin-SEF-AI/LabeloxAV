"""The settlement engine's daemon hooks: build lots, tally verdicts, settle, and keep checking.

Four self-guarding maybe_* hooks in the fleet idiom, each one stage of the lot lifecycle:

- **build**: one new lot per night, for the L1+ class whose judged precision is best - the VLM judge
  ORDERS which stratum to try first and never decides (Rogan-Gladen clamps on 9 of 13 measured
  classes, so humans are the only acceptance denominator). Evidence collection needs no autonomy:
  lots are built and judged at L1; only the settle WRITE is gated.
- **tally**: read each judging lot's human verdicts, run the acceptance decision, top up an
  inconclusive lot by its own samples-needed figure (at most twice). A passed lot IS the 1->2
  promotion for its class and epoch.
- **settle**: write the accepted lots whose class is at L2, behind `settlement_enabled` and the kill
  switch, chunked and revertible. Tier rules: default settles unattended; safety's FIRST lot per
  epoch waits for a one-click human ack (a notification, then the ack flips the lot loose); critical
  never auto-applies - settle_lot itself refuses it.
- **spot check**: nightly, run the acceptance decision IN REVERSE over each settled lot's spot
  verdicts. A reverse reject - the 95% lower bound on the spot defect rate above the lot's own FAR
  bound - auto-reverts the lot (objects to `review`, the conservative direction), steps the class to
  L0, audits, notifies. With the kill switch engaged even this revert becomes a one-click proposal:
  a person who pulled the cord gets no surprise writes in either direction.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import SettlementLot, SettlementSpot

log = get_logger("govern.settlement_agent")

BUILD_KIND = "settlement_build"
SPOT_KIND = "settlement_spot_check"
LOTS_PER_NIGHT = 1


async def _ranked_candidates(db: AsyncSession) -> list[dict]:
    """L1+ classes with no open lot, best judged precision first.

    The ordering signal is the class-precision batch's raw agreement rate - the judge's opinion, used
    only to decide what to TRY first. An unmeasured class still qualifies; it just goes last.
    """
    from services.autolabel.ontology import get_ontology
    from services.govern.class_autonomy import effective_level
    from services.labelops.class_precision import batch_id_for
    from services.labelops.settlement import MIN_POPULATION, strata_for_class
    from services.labelops.vlm_review import judged_precision

    onto = get_ontology()
    open_classes = set((await db.execute(select(SettlementLot.class_id).where(
        SettlementLot.status.in_(("sampling", "judging", "accepted"))))).scalars().all())

    out = []
    for cls in onto.classes:
        if cls.id in open_classes:
            continue
        lvl = await effective_level(db, cls.id)
        if lvl["level"] < 1:
            continue
        strata = await strata_for_class(db, cls.id, cls.name)
        if not strata or strata[0]["population"] < MIN_POPULATION:
            continue
        try:
            jp = await judged_precision(db, batch_id_for(cls.name))
            score = (jp.get("raw") or {}).get("p")
        except Exception:  # noqa: BLE001 - an unmeasured class is orderable, just last
            score = None
        out.append({"class_name": cls.name, "class_id": cls.id, "level": lvl["level"],
                    "population": strata[0]["population"], "epoch": strata[0]["epoch"],
                    "judged_precision": score})
    out.sort(key=lambda r: (r["judged_precision"] is None, -(r["judged_precision"] or 0.0)))
    return out


async def maybe_build_lots(db: AsyncSession) -> dict:
    """Off-hours hook: plan one lot per night for the best-ranked eligible class."""
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.govern.killswitch import get_state

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, BUILD_KIND, day_start):
        return {"ran": False, "reason": "already ran today"}
    state = await get_state(db)
    if not state.loop_enabled:
        return {"ran": False, "reason": "kill switch engaged"}

    ranked = await _ranked_candidates(db)
    if not ranked:
        return {"ran": False, "reason": "no L1+ class has an eligible stratum without an open lot"}
    picks = ranked[:LOTS_PER_NIGHT]

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.labelops.settlement import plan_lot
        from services.notify import notify

        report: dict = {"planned": [], "refused": []}
        status = "committed"
        try:
            async with get_sessionmaker()() as wdb:
                for cand in picks:
                    res = await plan_lot(wdb, cand["class_name"], epoch=cand["epoch"],
                                         created_by="settlement_agent")
                    if "error" in res:
                        report["refused"].append({"class_name": cand["class_name"],
                                                  "reason": res["error"]})
                        continue
                    report["planned"].append(res)
                    await notify(
                        wdb, kind="gate_batch_ready", severity="info",
                        title=f"settlement sample ready: {res['sample_n']} {cand['class_name']} "
                              f"crops (~{res['human_minutes_estimate']} min)",
                        body=(f"Verdicts on this sample decide whether {res['population']} "
                              f"{cand['class_name']} labels settle (far bound {res['far_bound']})."),
                        href=res["review_at"], subject_type="settlement_lot",
                        subject_id=res["lot_id"],
                        meta={k: res[k] for k in ("lot_id", "batch_id", "population", "tier")})
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("settlement.build_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, "candidates": [c["class_name"] for c in picks],
            **(await launch(db, BUILD_KIND, worker, created_by="scheduler"))}


async def _first_epoch_lot(db: AsyncSession, lot: SettlementLot) -> bool:
    n = (await db.execute(select(SettlementLot.lot_id).where(
        SettlementLot.class_id == lot.class_id, SettlementLot.model_epoch == lot.model_epoch,
        SettlementLot.status == "settled",
        SettlementLot.lot_id != lot.lot_id))).scalars().first()
    return n is None


async def maybe_tally_and_settle(db: AsyncSession) -> dict:
    """Every-tick hook (cheap when nothing moves): tally judging lots, settle what may settle.

    Runs inline rather than as a launched worker: a tally is a handful of reads, and settling commits
    chunk by chunk inside settle_lot, so the tick never holds a long transaction. Everything that can
    refuse does so with a reason that lands in the returned actions.
    """
    from services.autolabel.ontology import get_ontology
    from services.govern.class_autonomy import effective_level, promote_to_settlement
    from services.govern.killswitch import get_state
    from services.labelops.settlement import settle_lot, tally_lot, top_up_lot
    from services.notify import notify

    lots = (await db.execute(select(SettlementLot).where(
        SettlementLot.status.in_(("judging", "accepted"))))).scalars().all()
    if not lots:
        return {"ran": False, "reason": "no open lots"}

    onto = get_ontology()
    state = await get_state(db)
    actions = []
    for lot in lots:
        class_name = onto.by_id(lot.class_id).name
        if lot.status == "judging":
            res = await tally_lot(db, str(lot.lot_id))
            if res.get("status") == "judging" and (res.get("decision") or {}).get(
                    "verdict") == "inconclusive":
                res["topup"] = await top_up_lot(db, str(lot.lot_id))
            if res.get("status") == "accepted":
                # The passed lot IS the 1->2 promotion for this class+epoch.
                res["ladder"] = await promote_to_settlement(
                    db, lot.class_id, lot_id=str(lot.lot_id),
                    basis={"decision": res.get("decision"), "epoch": lot.model_epoch})
            if res.get("status") == "rejected":
                from services.govern.class_autonomy import step_down

                res["ladder"] = await step_down(db, lot.class_id, 0,
                                               reason=f"lot {lot.lot_id} rejected: "
                                                      f"{(res.get('decision') or {}).get('reason')}",
                                               set_by=f"lot:{lot.lot_id}")
                await notify(db, kind="gate_batch_ready", severity="warn",
                             title=f"settlement lot rejected for {class_name}",
                             body=(res.get("decision") or {}).get("reason"),
                             href="/govern", subject_type="settlement_lot",
                             subject_id=str(lot.lot_id), meta=res.get("decision") or {})
            if res.get("status") != "judging" or res.get("topup"):
                actions.append({"lot": str(lot.lot_id), "class_name": class_name, **{
                    k: res[k] for k in ("status", "sample_n", "defects") if k in res}})
            continue

        # status == accepted: settle if allowed
        if lot.tier == "critical":
            continue    # settle_lot refuses it anyway; no point asking nightly
        lvl = await effective_level(db, lot.class_id)
        if lvl["level"] < 2:
            continue
        if not state.settlement_enabled or not state.loop_enabled:
            continue    # the lot waits, visibly, in status accepted
        if lot.tier == "safety" and await _first_epoch_lot(db, lot):
            acked = bool((lot.decision or {}).get("human_ack"))
            if not acked:
                await notify(db, kind="gate_batch_ready", severity="warn",
                             title=f"first {class_name} settlement this epoch needs a one-click ack",
                             body=(f"Lot passed (far {lot.far_bound}); a safety class's first lot "
                                   "per epoch settles only after a person acks it."),
                             href="/govern", subject_type="settlement_ack",
                             subject_id=str(lot.lot_id),
                             meta={"lot_id": str(lot.lot_id), "tier": lot.tier})
                continue
        res = await settle_lot(db, str(lot.lot_id), created_by="settlement_agent")
        actions.append({"lot": str(lot.lot_id), "class_name": class_name, **{
            k: res[k] for k in ("status", "settled", "spots", "error") if k in res}})

    return {"ran": bool(actions), "actions": actions} if actions else {
        "ran": False, "reason": "open lots are waiting on verdicts, acks, or the operator switch"}


async def maybe_spot_check(db: AsyncSession) -> dict:
    """Off-hours hook: the reverse acceptance decision over every settled lot's spot verdicts.

    Reject here means the spot sample's defect rate is provably above the lot's own bound - the lot's
    acceptance was wrong about the population, whatever the reason - and the ONLY automatic remedy is
    the conservative one: everything the lot settled returns to review, the class steps to L0.
    """
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.labelops.sampling import acceptance_decision

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, SPOT_KIND, day_start):
        return {"ran": False, "reason": "already ran today"}

    lots = (await db.execute(select(SettlementLot).where(
        SettlementLot.status == "settled"))).scalars().all()
    if not lots:
        return {"ran": False, "reason": "nothing is settled"}

    checks = []
    for lot in lots:
        spots = (await db.execute(select(SettlementSpot).where(
            SettlementSpot.lot_id == lot.lot_id,
            SettlementSpot.human_verdict.isnot(None)))).scalars().all()
        defects = sum(1 for s in spots if s.human_verdict == "incorrect")
        lot.spot_defects = defects
        decision = acceptance_decision(defects, len(spots), max_defect_rate=lot.far_bound)
        checks.append({"lot": lot, "n": len(spots), "defects": defects, "decision": decision})
    await db.commit()

    breaches = [c for c in checks if c["decision"]["verdict"] == "reject"]
    if not breaches:
        return {"ran": False,
                "reason": f"{len(checks)} settled lots spot-checked; none breach their bound"}

    from services.govern.killswitch import get_state

    state = await get_state(db)
    engaged = not state.loop_enabled
    breach_info = [{"lot_id": str(c["lot"].lot_id), "class_id": c["lot"].class_id,
                    "n": c["n"], "defects": c["defects"], "reason": c["decision"]["reason"]}
                   for c in breaches]

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.govern.class_autonomy import step_down
        from services.labelops.settlement import revert_lot
        from services.notify import notify

        onto = get_ontology()
        report: dict = {"breaches": breach_info, "reverted": [], "proposed": []}
        status = "committed"
        try:
            async with get_sessionmaker()() as wdb:
                for b in breach_info:
                    cname = onto.by_id(b["class_id"]).name
                    if engaged:
                        # The cord is pulled: no writes, one click instead.
                        report["proposed"].append(b["lot_id"])
                        await notify(wdb, kind="gate_batch_ready", severity="error",
                                     title=f"spot check failed on settled {cname} lot; revert "
                                           "awaits a click (kill switch engaged)",
                                     body=b["reason"], href="/govern",
                                     subject_type="settlement_lot", subject_id=b["lot_id"],
                                     meta=b)
                        continue
                    rev = await revert_lot(wdb, b["lot_id"],
                                           reason=f"spot check reject: {b['reason']}")
                    await step_down(wdb, b["class_id"], 0,
                                    reason=f"spot check reject on lot {b['lot_id']}",
                                    set_by="spot_check")
                    report["reverted"].append({"lot_id": b["lot_id"], **{
                        k: rev[k] for k in ("reverted", "skipped", "error") if k in rev}})
                    await notify(wdb, kind="gate_batch_ready", severity="error",
                                 title=f"settled {cname} lot auto-reverted: spot check failed",
                                 body=b["reason"], href="/govern",
                                 subject_type="settlement_lot", subject_id=b["lot_id"], meta=b)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("settlement.spot_check_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, "breaches": len(breaches),
            **(await launch(db, SPOT_KIND, worker, created_by="scheduler"))}
