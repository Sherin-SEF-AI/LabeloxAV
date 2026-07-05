"""The SANYX quarantine gate: the mechanism by which a bad session never costs a label. Downstream planes
(SIEVYX curation, the label queue, ORACLYX) consult this before doing any work on a session's samples. The
decision lives on the session's latest SessionHealth row (score + decision), written by run.py.

A session with no health report yet is treated as ungated (unknown), not blocked, so ingestion is not a
hard dependency on SANYX having run; the orchestration DAG (M9) is what enforces "run SANYX first". This
function answers the narrower question "is this session known-bad".
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import SessionHealth


async def latest_health(db: AsyncSession, session_id) -> SessionHealth | None:
    return (await db.execute(
        select(SessionHealth).where(SessionHealth.session_id == session_id)
        .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()


async def is_quarantined(db: AsyncSession, session_id) -> bool:
    """True only when the latest health report decided quarantine. Unknown (no report) is not quarantined."""
    h = await latest_health(db, session_id)
    return bool(h and (h.decision == "quarantine" or (h.decision is None and h.verdict == "fail")))


async def gate_state(db: AsyncSession, session_id) -> dict:
    """The gate view for a session: decision, score, and whether downstream may proceed."""
    h = await latest_health(db, session_id)
    if h is None:
        return {"decision": None, "score": None, "may_proceed": True, "reason": "no health report yet"}
    decision = h.decision or {"pass": "pass", "warn": "degraded", "fail": "quarantine"}.get(h.verdict, h.verdict)
    return {
        "decision": decision,
        "score": h.score,
        "may_proceed": decision != "quarantine",
        "reason": "quarantined by SANYX" if decision == "quarantine" else "cleared",
    }
