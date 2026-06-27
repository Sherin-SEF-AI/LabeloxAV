"""Governance endpoints (M4.4): model registry + champion/challenger promotion, control-sample precision,
drift scan, the controller tick, the kill switch, and the audit log."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.govern import killswitch as K
from services.govern.audit import list_audit
from services.govern.champion import evaluate_and_promote
from services.govern.control_sample import measured_precision, record_verdict, seed_from_recent_auto_accepts
from services.govern.controller import tick
from services.govern.drift import run_drift_scan
from services.govern.registry import list_models, register, register_from_run

router = APIRouter()


# ---- registry + promotion ----
class RegisterIn(BaseModel):
    model_version: str
    task: str = "detection"
    gold_metrics: dict
    dataset_commit: str | None = None
    weights_uri: str | None = None
    notes: str | None = None


@router.post("/govern/registry/register")
async def registry_register(payload: RegisterIn, db: AsyncSession = Depends(db_session)):
    return await register(db, payload.model_version, payload.task, payload.gold_metrics,
                          payload.dataset_commit, payload.weights_uri, payload.notes)


@router.post("/govern/registry/register_run")
async def registry_register_run(run_id: str, task: str | None = None, db: AsyncSession = Depends(db_session)):
    return await register_from_run(db, run_id, task)


@router.get("/govern/registry")
async def registry_list(task: str | None = None, db: AsyncSession = Depends(db_session)):
    return await list_models(db, task)


@router.post("/govern/promote")
async def promote(model_version: str, task: str = "detection", db: AsyncSession = Depends(db_session)):
    return await evaluate_and_promote(db, model_version, task)


# ---- control sample ----
@router.post("/govern/control/seed")
async def control_seed(limit: int = 500, rate: float | None = None, db: AsyncSession = Depends(db_session)):
    return await seed_from_recent_auto_accepts(db, limit, rate)


class VerdictIn(BaseModel):
    verdict: str


@router.post("/govern/control/{sample_id}/verdict")
async def control_verdict(sample_id: str, payload: VerdictIn, db: AsyncSession = Depends(db_session)):
    return await record_verdict(db, sample_id, payload.verdict)


@router.get("/govern/control/precision")
async def control_precision(db: AsyncSession = Depends(db_session)):
    return await measured_precision(db)


# ---- drift + controller ----
class DriftIn(BaseModel):
    ref_sessions: list[str] | None = None
    cur_sessions: list[str] | None = None


@router.post("/govern/drift/scan")
async def drift_scan(payload: DriftIn, db: AsyncSession = Depends(db_session)):
    return await run_drift_scan(db, payload.ref_sessions, payload.cur_sessions)


@router.post("/govern/controller/tick")
async def controller_tick(schedule_bursts: bool = True, db: AsyncSession = Depends(db_session)):
    return await tick(db, schedule_bursts=schedule_bursts)


# ---- kill switch + state + audit ----
@router.get("/govern/state")
async def state(db: AsyncSession = Depends(db_session)):
    return await K.state_dict(db)


class EngageIn(BaseModel):
    reason: str = "manual kill switch"
    task: str = "detection"


@router.post("/govern/killswitch/engage")
async def killswitch_engage(payload: EngageIn, db: AsyncSession = Depends(db_session)):
    return await K.engage(db, payload.reason, payload.task)


@router.post("/govern/killswitch/release")
async def killswitch_release(db: AsyncSession = Depends(db_session)):
    return await K.release(db)


@router.get("/govern/audit")
async def audit(actor: str | None = None, limit: int = 100, db: AsyncSession = Depends(db_session)):
    return await list_audit(db, actor, limit)
