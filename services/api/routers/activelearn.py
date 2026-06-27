"""Active-learning endpoints (M4.0): score the candidate pool, select a budgeted batch, list batches,
and drive the retrain-selection loop."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AlSelection
from services.activelearn.budget import select_batch
from services.activelearn.loop import maybe_retrain, new_signal_count
from services.activelearn.selector import score_candidates
from services.api.deps import db_session

router = APIRouter()


class SelectIn(BaseModel):
    budget_hours: float = 1.0
    session_id: str | None = None
    dedup_cos: float = 0.92


@router.get("/activelearn/score")
async def score(session_id: str | None = None, limit: int = 50, db: AsyncSession = Depends(db_session)):
    items = await score_candidates(db, session_id)
    return {"pool": len(items), "items": items[:limit]}


@router.post("/activelearn/select")
async def select(payload: SelectIn, db: AsyncSession = Depends(db_session)):
    return await select_batch(db, payload.budget_hours, payload.session_id, payload.dedup_cos)


@router.get("/activelearn/batches")
async def batches(db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(AlSelection).order_by(AlSelection.created_at.desc()).limit(50))).scalars().all()
    return [{"batch_id": str(b.batch_id), "n_items": len(b.item_ids or []), "budget_hours": b.budget_hours,
             "status": b.status, "expected_value": b.expected_value, "strategy": b.strategy,
             "created_at": b.created_at.isoformat() if b.created_at else None} for b in rows]


@router.get("/activelearn/loop")
async def loop_status(db: AsyncSession = Depends(db_session)):
    return await new_signal_count(db)


class RetrainIn(BaseModel):
    compute_target: str = "cloud"
    force: bool = False
    base_weights: str | None = None


@router.post("/activelearn/loop/retrain")
async def loop_retrain(payload: RetrainIn, db: AsyncSession = Depends(db_session)):
    return await maybe_retrain(db, payload.compute_target, payload.force, payload.base_weights)
