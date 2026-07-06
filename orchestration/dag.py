"""The flywheel DAG (M9). A session flows through every plane automatically, and the upstream gates are
enforced here: a session SANYX quarantined or CALYX blocked never reaches SIEVYX or the label queue, which is
the mechanism by which bad data never costs a label. The stages are ordered by the platform registry's
flywheel_stage, so adding a plane reorders the loop by data, not by editing this file.

plan() is pure over the gate outcomes so the progression logic (stops at the failing gate) is unit-testable;
run_session() wires it to the live gate state, and trace_lineage() resolves a deployed model all the way back
to its source sessions and their health and calibration state (the one-spine query the engine exists for)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from platforms.registry import PLATFORMS

log = get_logger("flywheel")

# the flywheel stages in loop order (annotation core sits between curation and fusion)
STAGES = [p.id for p in sorted((p for p in PLATFORMS if p.flywheel_stage is not None),
                               key=lambda p: p.flywheel_stage)]


def plan(sanyx_decision: str | None, calyx_severity: str | None) -> dict:
    """Decide how far a session progresses given the two blocking gates. A quarantine (SANYX) or block
    (CALYX) stops the session before curation, so no label is spent on it."""
    if sanyx_decision == "quarantine":
        return {"proceed": False, "stopped_at": "sanyx", "reason": "session quarantined by SANYX",
                "stages_run": ["sanyx"]}
    if calyx_severity == "block":
        return {"proceed": False, "stopped_at": "calyx", "reason": "session calibration blocked by CALYX",
                "stages_run": ["sanyx", "calyx"]}
    return {"proceed": True, "stopped_at": None, "reason": "cleared both gates",
            "stages_run": STAGES}


async def run_session(db: AsyncSession, session_id) -> dict:
    """Run the flywheel for one session: consult the SANYX and CALYX gates, and if cleared, report the
    downstream stages that proceed. Returns the progression trace."""
    from db.models import CalibrationValidation, SessionHealth

    health = (await db.execute(
        select(SessionHealth).where(SessionHealth.session_id == session_id)
        .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()
    sanyx_decision = (health.decision or {"pass": "pass", "warn": "degraded", "fail": "quarantine"}
                      .get(health.verdict)) if health else None
    calyx_sev = (await db.execute(
        select(CalibrationValidation.severity).where(CalibrationValidation.session_id == session_id,
                                                     CalibrationValidation.severity.isnot(None))
        .order_by(CalibrationValidation.created_at.desc()).limit(1))).scalar_one_or_none()

    p = plan(sanyx_decision, calyx_sev)
    log.info("flywheel.session", session=str(session_id), proceed=p["proceed"], stopped_at=p["stopped_at"])
    return {"session_id": str(session_id), "sanyx_decision": sanyx_decision, "calyx_severity": calyx_sev, **p}


async def trace_lineage(db: AsyncSession, deployment_id) -> dict:
    """Resolve a deployed model back to its release, its VERDYX verdict, its FORGYX benchmark, and the exact
    source sessions with their SANYX health and CALYX calibration state. One query, whole loop."""
    from db.models import Deployment, Evaluation, GoldSet
    from services.release.lineage import resolve_lineage

    dep = await db.get(Deployment, deployment_id)
    if dep is None:
        return {"error": "deployment not found"}
    verdict = await db.get(Evaluation, dep.verdict_ref) if dep.verdict_ref else None

    # resolve the release members to sessions: a gold-set-backed eval carries the exact object ids
    object_ids: list[str] = []
    if verdict is not None and verdict.gold_id:
        gs = await db.get(GoldSet, verdict.gold_id)
        object_ids = [str(o) for o in (gs.object_ids or [])] if gs else []
    lineage = await resolve_lineage(db, object_ids) if object_ids else {"sessions": [], "n_sessions": 0}

    return {
        "deployment_id": str(deployment_id), "model_version": dep.model_version, "target": dep.target,
        "status": dep.status, "release_commit": dep.release_commit,
        "verdict": verdict.verdict if verdict else None,
        "source_sessions": lineage.get("sessions", []),
        "flagged_sessions": lineage.get("flagged_sessions", []),
    }
