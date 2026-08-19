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

    # continuous embedder: every tick, drain a bounded slice of the unembedded backlog so find-similar
    # coverage tracks new data instead of drifting to zero. Not gated to off-hours (coverage should stay
    # current all day); it self-guards on free VRAM, so when a detector or a training run holds the card it
    # yields this tick rather than competing.
    try:
        from services.intelligence.embed.daemon import maybe_embed_pending

        e = await maybe_embed_pending(db)
        if e.get("ran"):
            actions.append({"action": "embed_pending", "frames": e.get("embedded_frames"),
                            "objects": e.get("embedded_objects"), "remaining": e.get("pending_after")})
    except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
        log.error("schedule.embed_daemon_failed", error=str(exc))

    # nightly rebuild of the class-compatibility matrix. It is a corpus-wide aggregate over
    # human-confirmed co-occurrence, so it moves only when somebody labels, and reading it per frame would
    # be a full-corpus scan on the labelling hot path. Off-hours because it walks every human object.
    if offhours:
        try:
            from services.autolabel.compat_matrix import maybe_rebuild_matrix

            c = await maybe_rebuild_matrix(db)
            if c.get("ran"):
                actions.append({"action": "compat_matrix", "cells": c.get("n_cells_observed"),
                                "observations": c.get("n_observations"),
                                "objects": c.get("n_objects"), "learned": c.get("learned")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.compat_matrix_failed", error=str(exc))

    # nightly patrol of the day's auto-accepts
    if offhours:
        try:
            from services.agent.overnight_auditor import maybe_run_nightly

            a = await maybe_run_nightly(db)
            if a.get("ran"):
                actions.append({"action": "overnight_audit", "run_id": a.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.auditor_failed", error=str(exc))

        # nightly adaptive flywheel: refresh the label/collect plan and the fleet collection orders from the
        # live corpus, so the worklist and the dispatch board are current each morning. It proposes only (runs
        # the cycle and the collection orders); the human still clicks send-to-review and dispatches drives.
        try:
            from services.flywheel.auto_schedule import maybe_run_flywheel

            f = await maybe_run_flywheel(db)
            if f.get("ran"):
                actions.append({"action": "flywheel_cycle", "cycle_id": f.get("cycle_id"),
                                "orders": f.get("orders")})
        except Exception as exc:  # noqa: BLE001
            log.error("schedule.flywheel_failed", error=str(exc))

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
