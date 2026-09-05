"""One endpoint that answers "what is the machine allowed to do right now, and why".

Everything on this page existed as scattered truth - the switches on governance_state, the ladder in
class_autonomy, staleness inside each measurement's own rows, the night's story across AgentRuns -
and reading it meant knowing where each piece lived. The whole measurement stack had zero web callers
until phase 3; this is the aggregation those surfaces read.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()

# The daemon ticks every 60s and records an audit row each time; two missed ticks is "stale", shown,
# not guessed. A dead daemon looks exactly like a healthy idle one on every other surface.
TICK_SECONDS = 60
STALE_AFTER_TICKS = 2
# A tick that is still running has only its start marker, and a full challenger sweep on the GPU
# can hold a tick open for tens of minutes. Mid-tick is alive up to this bound; past it, a start
# with no completion is a daemon that died mid-work, which IS stale.
TICK_MAX_SECONDS = 3600


async def _daemon_liveness(db: AsyncSession) -> dict:
    from db.models import AuditDecision

    row = (await db.execute(
        select(AuditDecision.decision, AuditDecision.created_at)
        .where(AuditDecision.actor == "controller",
               AuditDecision.decision.in_(("tick", "tick_paused", "tick_started")))
        .order_by(AuditDecision.created_at.desc()).limit(1))).first()
    if row is None:
        return {"alive": False, "last_tick_at": None, "seconds_since": None,
                "detail": "no tick has ever been recorded; the daemon has not run"}
    age = (datetime.now(UTC) - row.created_at).total_seconds()
    if row.decision == "tick_started":
        # Newest marker is a start with no completion: the tick is still running. Alive, and SAYS
        # it is mid-tick, up to the bound past which an unfinished start means it died mid-work.
        return {"alive": age <= TICK_MAX_SECONDS,
                "last_tick_at": row.created_at.isoformat(), "seconds_since": round(age),
                "last_status": f"mid-tick (started {round(age)}s ago)"}
    return {"alive": age <= TICK_SECONDS * STALE_AFTER_TICKS,
            "last_tick_at": row.created_at.isoformat(), "seconds_since": round(age),
            "last_status": row.decision}


async def _measurement_staleness(db: AsyncSession) -> dict:
    from db.models import MachineVerdict
    from services.govern.control_sample import measured_precision
    from services.labelops.class_precision import BATCH_PREFIX

    prec = await measured_precision(db)

    per_class = (await db.execute(
        select(MachineVerdict.batch_id, func.max(MachineVerdict.created_at),
               func.count(MachineVerdict.verdict_id))
        .where(MachineVerdict.batch_id.startswith(f"{BATCH_PREFIX}:"))
        .group_by(MachineVerdict.batch_id))).all()
    now = datetime.now(UTC)
    class_rows = sorted(
        ({"class_name": b.split(":", 1)[1], "measured_at": ts.isoformat(),
          "age_days": round((now - ts).total_seconds() / 86400, 1), "verdicts": int(n)}
         for b, ts, n in per_class),
        key=lambda r: -r["age_days"])

    cal = (await db.execute(
        select(func.max(MachineVerdict.created_at), func.count(MachineVerdict.verdict_id))
        .where(MachineVerdict.batch_id == "judge-calibration"))).first()
    calibration = {"measured_at": cal[0].isoformat() if cal and cal[0] else None,
                   "verdicts": int(cal[1]) if cal else 0,
                   "age_days": (round((now - cal[0]).total_seconds() / 86400, 1)
                                if cal and cal[0] else None)}

    from db.models import ThresholdFit

    fits = (await db.execute(
        select(func.count()).select_from(ThresholdFit)
        .where(ThresholdFit.active.is_(True), ThresholdFit.measured.is_(True)))).scalar_one()

    return {"control_precision": prec, "class_precision": class_rows,
            "judge_calibration": calibration, "active_fitted_thresholds": int(fits)}


async def _settlement_summary(db: AsyncSession) -> dict:
    from db.models import AgentRun, SettlementLot

    by_status = {k: int(v) for k, v in (await db.execute(
        select(SettlementLot.status, func.count()).group_by(SettlementLot.status))).all()}
    from db.models import Object

    settled_objects = (await db.execute(
        select(func.count()).select_from(Object)
        .where(Object.state == "settled"))).scalar_one()
    runs = (await db.execute(
        select(AgentRun.status, func.count()).where(AgentRun.kind == "settlement")
        .group_by(AgentRun.status))).all()
    run_counts = {k: int(v) for k, v in runs}
    total_runs = sum(run_counts.values())
    return {"lots_by_status": by_status, "settled_objects": int(settled_objects),
            "settlement_runs": run_counts,
            # The displayed control signal: how often settlement has had to be taken back.
            "revert_rate": (round(run_counts.get("reverted", 0) / total_runs, 3)
                            if total_runs else None)}


@router.get("/autonomy/state", dependencies=[Depends(require_role("annotator"))])
async def autonomy_state(db: AsyncSession = Depends(db_session)):
    from db.models import AgentRun
    from services.agent.runtime.report import latest_run
    from services.govern.class_autonomy import ladder_snapshot
    from services.govern.killswitch import get_state

    st = await get_state(db)
    journal = (await db.execute(
        select(AgentRun.kind, AgentRun.status, AgentRun.created_at, AgentRun.run_id)
        .order_by(AgentRun.created_at.desc()).limit(25))).all()

    return {
        "switches": {"loop_enabled": st.loop_enabled, "auto_accept_enabled": st.auto_accept_enabled,
                     "auto_promote_enabled": st.auto_promote_enabled,
                     "settlement_enabled": st.settlement_enabled,
                     "paused_reason": st.paused_reason, "champion_version": st.champion_version},
        "daemon": await _daemon_liveness(db),
        "ladder": await ladder_snapshot(db),
        "measurements": await _measurement_staleness(db),
        "settlement": await _settlement_summary(db),
        "last_digest": await latest_run(db, "nightly_digest"),
        "journal": [{"kind": k, "status": s2, "created_at": c.isoformat() if c else None,
                     "run_id": str(r)} for k, s2, c, r in journal],
    }
