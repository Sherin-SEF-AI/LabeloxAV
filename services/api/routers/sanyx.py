"""SANYX ingest-QA endpoints: the ingest board (sessions with their latest health), a per-session report with
per-check evidence, on-demand health runs, the quarantine gate view, and a quarantine override with reason
capture. Mounted under /api; the platform registry maps /sanyx (and the legacy /inspector, /adverse) here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Session as DbSession
from db.models import SessionHealth
from services.api.deps import db_session
from services.sanyx.gate import gate_state, latest_health
from services.sanyx.run import run_health

router = APIRouter()


def _report(h: SessionHealth | None) -> dict | None:
    if h is None:
        return None
    return {
        "session_id": str(h.session_id), "score": h.score,
        "decision": h.decision or {"pass": "pass", "warn": "degraded", "fail": "quarantine"}.get(h.verdict),
        "verdict": h.verdict, "checks": h.checks, "indexer_version": h.indexer_version,
        "created_at": h.created_at.isoformat() if h.created_at else None,
    }


@router.get("/sanyx/board")
async def board(limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The ingest board: recent sessions with their latest SANYX health score and decision."""
    limit = min(max(limit, 1), 500)
    sessions = (await db.execute(
        select(DbSession).order_by(DbSession.created_at.desc()).limit(limit))).scalars().all()
    rows = []
    for s in sessions:
        h = await latest_health(db, s.session_id)
        rep = _report(h)
        rows.append({
            "session_id": str(s.session_id), "vehicle_id": s.vehicle_id, "city": s.city,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "score": rep["score"] if rep else None,
            "decision": rep["decision"] if rep else None,
            "n_checks": len(rep["checks"]) if rep else 0,
        })
    return {"sessions": rows}


@router.get("/sanyx/session/{session_id}")
async def session_report(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    rep = _report(await latest_health(db, session_id))
    if rep is None:
        raise HTTPException(404, "no health report for this session (run SANYX first)")
    return rep


@router.post("/sanyx/session/{session_id}/run")
async def run(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Run the six checks on demand and persist a fresh HealthReport."""
    if await db.get(DbSession, session_id) is None:
        raise HTTPException(404, "session not found")
    return await run_health(session_id)


@router.get("/sanyx/gate/{session_id}")
async def gate(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """The quarantine gate view downstream planes consult before spending work on a session."""
    return await gate_state(db, session_id)


class OverrideIn(BaseModel):
    decision: str  # pass | degraded | quarantine
    reason: str


@router.post("/sanyx/session/{session_id}/override")
async def override(session_id: uuid.UUID, payload: OverrideIn, db: AsyncSession = Depends(db_session)):
    """Human override of a SANYX decision, with a captured reason. Writes a new health row so the override is
    auditable and the latest decision reflects the human call, without mutating the machine report."""
    if payload.decision not in ("pass", "degraded", "quarantine"):
        raise HTTPException(400, "decision must be pass, degraded, or quarantine")
    prior = await latest_health(db, session_id)
    if prior is None:
        raise HTTPException(404, "no health report to override (run SANYX first)")
    verdict = {"pass": "pass", "degraded": "warn", "quarantine": "fail"}[payload.decision]
    checks = list(prior.checks or []) + [{
        "name": "human_override", "status": payload.decision, "score": None,
        "detail": payload.reason, "evidence": {"overrides": prior.decision or prior.verdict}}]
    row = SessionHealth(session_id=session_id, checks=checks, verdict=verdict, score=prior.score,
                        decision=payload.decision, indexer_version="sanyx-override")
    db.add(row)
    await db.commit()
    return _report(row)
