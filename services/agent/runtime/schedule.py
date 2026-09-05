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

    # standing campaigns: advance each one by at most one stage per tick. The state machine self-guards
    # (waits on humans at the label stage, on the GPU lease at train, on approvals unless a stage is on
    # autopilot), so ticking is cheap when nothing can move and this is what makes campaigns autonomous
    # rather than autonomous-while-watched.
    try:
        from services.flywheel.campaign import maybe_tick_campaigns

        t = await maybe_tick_campaigns(db)
        if t.get("ran"):
            actions.append({"action": "campaign_tick", "ticked": t.get("ticked")})
    except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
        log.error("schedule.campaign_tick_failed", error=str(exc))

    # settlement lifecycle, the cheap half: tally judging lots as verdicts arrive, settle accepted
    # ones the guards allow. Inline (no worker): a tally is a handful of reads and settle_lot commits
    # chunk by chunk itself.
    try:
        from services.govern.settlement_agent import maybe_tally_and_settle

        ts = await maybe_tally_and_settle(db)
        if ts.get("ran"):
            actions.append({"action": "settlement_tick", "lots": ts.get("actions")})
    except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
        log.error("schedule.settlement_tick_failed", error=str(exc))

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

    # Per-frame rarity onto Frame.scene, so a slice, an export and the coverage datasheet can all select on
    # how unusual a frame is. Off-hours and bounded: rarity moves as labelling moves, so this is a sweep
    # that catches up rather than a backfill that finishes.
    if offhours:
        try:
            from services.context.rarity import sweep_rarity

            r = await sweep_rarity(db, limit=4000)
            if r.get("scored"):
                actions.append({"action": "rarity_sweep", "scored": r["scored"],
                                "remaining": r.get("remaining")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.rarity_sweep_failed", error=str(exc))

    # a blocked promotion fires its own unblock attempt: VLM re-review of the starved classes first
    # (free, revertible), then a gate-directed review batch for the remaining deficit, then a
    # notification. Once per blocked run - a retrain moves the deficit, not a second scan of the pool.
    if offhours:
        try:
            from services.agent.gate_unblock import maybe_unblock_gate

            u = await maybe_unblock_gate(db)
            if u.get("ran"):
                actions.append({"action": "gate_unblock", "run_id": u.get("run_id"),
                                "target_run": u.get("target_run")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.gate_unblock_failed", error=str(exc))

    # nightly champion-degradation check against the sealed gold sets. This existed behind a button
    # (POST /api/agent/gold-drift) since the daemon work; a safety check that runs only when somebody
    # remembers to press it is a dashboard, not a check. Its rollback remedy is check_gold_drift's own.
    if offhours:
        try:
            from services.agent.training_daemon import maybe_check_gold_drift

            g = await maybe_check_gold_drift(db)
            if g.get("ran"):
                actions.append({"action": "gold_drift_check", "run_id": g.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.gold_drift_failed", error=str(exc))

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

    # measurement refresh: the denominators autonomy decides against, kept current instead of measured
    # once. Each maybe_* declines with a reason when its evidence has not moved or the card is busy.
    if offhours:
        try:
            from services.agent.measurement_agent import (
                maybe_judge_detectors,
                maybe_refresh_class_precision,
                maybe_refresh_judge_calibration,
            )

            for fn, label in ((maybe_refresh_class_precision, "class_precision_refresh"),
                              (maybe_refresh_judge_calibration, "judge_calibration_refresh"),
                              (maybe_judge_detectors, "detector_judging")):
                m = await fn(db)
                if m.get("ran"):
                    actions.append({"action": label, "run_id": m.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.measurement_failed", error=str(exc))

    # settlement lifecycle, the nightly half: plan one lot for the best-ranked eligible class, and run
    # the reverse acceptance decision over every settled lot's spot verdicts (the one automatic revert).
    if offhours:
        try:
            from services.govern.settlement_agent import maybe_build_lots, maybe_spot_check

            for fn, label in ((maybe_build_lots, "settlement_build"),
                              (maybe_spot_check, "settlement_spot_check")):
                sres = await fn(db)
                if sres.get("ran"):
                    actions.append({"action": label, "run_id": sres.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.settlement_nightly_failed", error=str(exc))

    # yardstick upkeep: reseal rotted gold sets (weekly), keep unfinished blind audits visible (daily).
    if offhours:
        try:
            from services.agent.yardstick_agent import maybe_nudge_blind_audit, maybe_repair_gold

            for fn, label in ((maybe_repair_gold, "gold_repair"),
                              (maybe_nudge_blind_audit, "blind_audit_nudge")):
                y = await fn(db)
                if y.get("ran"):
                    actions.append({"action": label, "run_id": y.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.yardstick_failed", error=str(exc))

    # the morning digest and the weekly report, last on purpose: the digest declines to send while any
    # agent run is still in flight, so putting it after every launcher gives tonight's agents their turn
    # before the first attempt rather than always deferring to tomorrow.
    if offhours:
        try:
            from services.agent.digest import maybe_send_digest, maybe_weekly_report

            w = await maybe_weekly_report(db)
            if w.get("ran"):
                actions.append({"action": "weekly_report", "run_id": w.get("run_id")})
            g2 = await maybe_send_digest(db)
            if g2.get("ran"):
                actions.append({"action": "nightly_digest", "run_id": g2.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - a fleet agent never blocks the governance loop
            log.error("schedule.digest_failed", error=str(exc))

    return actions
