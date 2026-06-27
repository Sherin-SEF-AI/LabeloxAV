"""Audit trail (M4.4): every automated decision is recorded, replayable, and visible. This is what makes
an unattended loop safe to run and what a buyer's diligence requires. Deterministic inputs in, one row
out, no judgement."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AuditDecision

log = get_logger("govern_audit")


async def record(db: AsyncSession, actor: str, decision: str, subject: str | None = None,
                 rationale: dict | None = None, commit: bool = True) -> str:
    a = AuditDecision(actor=actor, decision=decision, subject=subject, rationale=rationale or {})
    db.add(a)
    await db.flush()
    aid = str(a.audit_id)
    if commit:
        await db.commit()
    log.info("govern.audit", actor=actor, decision=decision, subject=subject)
    return aid


async def list_audit(db: AsyncSession, actor: str | None = None, limit: int = 100) -> list[dict]:
    q = select(AuditDecision).order_by(AuditDecision.created_at.desc()).limit(limit)
    if actor:
        q = q.where(AuditDecision.actor == actor)
    rows = (await db.execute(q)).scalars().all()
    return [{"audit_id": str(a.audit_id), "actor": a.actor, "decision": a.decision, "subject": a.subject,
             "rationale": a.rationale, "created_at": a.created_at.isoformat() if a.created_at else None}
            for a in rows]
