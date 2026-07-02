"""The fleet scheduler: the single place controller.tick calls to fire whichever agents are due this tick.

Each agent owns its own due-check (a maybe_* that returns {ran, ...} and self-guards against re-firing), so
the runtime stays a thin dispatcher: it just calls the right agents at the right trigger (off-hours cadence,
or on a drift breach). Adding a scheduled agent is one entry here plus its maybe_* function -- the "one
runtime, many agents" rule, kept minimal.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("agent.runtime.schedule")


async def run_due(db: AsyncSession, *, offhours: bool, drift: dict | None = None) -> list[dict]:
    actions: list[dict] = []

    # nightly patrol of the day's auto-accepts
    if offhours:
        try:
            from services.agent.overnight_auditor import maybe_run_nightly

            a = await maybe_run_nightly(db)
            if a.get("ran"):
                actions.append({"action": "overnight_audit", "run_id": a.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.auditor_failed", error=str(exc))

    # on-breach root-cause investigation
    if drift and drift.get("breached"):
        try:
            from services.agent.drift_investigator import maybe_investigate

            d = await maybe_investigate(db, drift)
            if d.get("ran"):
                actions.append({"action": "drift_investigation", "run_id": d.get("run_id")})
        except Exception as exc:  # noqa: BLE001
            log.error("schedule.drift_investigator_failed", error=str(exc))

    return actions
