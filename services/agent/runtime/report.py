"""Shared agent-report plumbing: launch a background agent run, finish it with a report + audit entry, and
read the latest report of a kind. Extracted from the Overnight Auditor once the Drift Investigator became a
second tenant, so every fleet agent records its work the same way (one AgentRun + one AuditDecision).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun
from db.session import get_sessionmaker


async def launch(db: AsyncSession, kind: str, worker: Callable[[uuid.UUID], Awaitable[None]], *,
                 created_by: str, policy: dict | None = None) -> dict:
    """Create the AgentRun row and fire the background worker; return the run id immediately."""
    run_id = uuid.uuid4()
    db.add(AgentRun(run_id=run_id, kind=kind, scope={}, status="running", policy=policy or {}, counts={},
                    changes={}, critic={}, created_by=created_by))
    await db.commit()
    asyncio.create_task(worker(run_id))
    return {"run_id": str(run_id), "status": "running"}


async def finish_run(run_id: uuid.UUID, *, status: str, report: dict, changes: dict | None = None,
                     decision: str = "report") -> None:
    """Persist the report onto the AgentRun and mirror it into the audit trail (actor = the run's kind)."""
    maker = get_sessionmaker()
    async with maker() as db:
        run = await db.get(AgentRun, run_id)
        actor = run.kind if run else "agent"
        if run:
            run.status = status
            run.counts = report
            if changes is not None:
                run.changes = changes
            await db.commit()
        try:
            from services.govern.audit import record

            await record(db, actor=actor, decision=decision, subject=str(run_id), rationale=report)
        except Exception:  # noqa: BLE001 - the audit mirror is best-effort; never fail the run over it
            pass


async def latest_run(db: AsyncSession, kind: str) -> dict | None:
    run = (await db.execute(select(AgentRun).where(AgentRun.kind == kind)
                            .order_by(AgentRun.created_at.desc()).limit(1))).scalar_one_or_none()
    if run is None:
        return None
    return {"run_id": str(run.run_id), "status": run.status,
            "created_at": run.created_at.isoformat() if run.created_at else None, "report": run.counts}


async def ran_since(db: AsyncSession, kind: str, since) -> bool:
    """True if an agent of this kind has a run created at/after `since` (the once-per-window marker)."""
    return (await db.execute(select(AgentRun.run_id).where(
        AgentRun.kind == kind, AgentRun.created_at >= since).limit(1))).first() is not None
