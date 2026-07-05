"""VERDYX slice-evaluation endpoints: record a per-slice evaluation with its champion-challenger verdict, and
read the slice matrix. The metrics come from the existing services/training/eval and Safe-mIoU; the champion
gate stays in services/govern/champion and consumes these verdicts. Mounted under /api; the registry groups
/training, /govern, /models under VERDYX."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Evaluation
from services.api.deps import db_session
from services.verdyx.run import matrix, record_evaluation

router = APIRouter()


class EvalIn(BaseModel):
    model_version: str
    aggregate: dict                 # {map50, map, precision, recall, safe_miou}
    per_slice: dict                 # {slice_id: {map, precision, recall}}
    challenger_of: str | None = None
    release_commit: str | None = None
    gold_id: str | None = None
    protected: list[str] | None = None


@router.post("/verdyx/evaluate")
async def evaluate(payload: EvalIn, db: AsyncSession = Depends(db_session)):
    """Record a per-slice evaluation and compute the protected-slice verdict governance consumes."""
    return await record_evaluation(db, payload.model_version, payload.aggregate, payload.per_slice,
                                   challenger_of=payload.challenger_of, release_commit=payload.release_commit,
                                   gold_id=payload.gold_id, protected=payload.protected)


@router.get("/verdyx/matrix")
async def slice_matrix_view(champion: str, challenger: str, slice_metric: str = "map",
                            db: AsyncSession = Depends(db_session)):
    """The slice matrix: champion vs challenger per slice, colored only by regression state in the UI."""
    return await matrix(db, champion, challenger, slice_metric)


@router.get("/verdyx/model/{model_version}/evals")
async def model_evals(model_version: str, limit: int = 20, db: AsyncSession = Depends(db_session)):
    """Per-model evaluation history."""
    rows = (await db.execute(
        select(Evaluation).where(Evaluation.model_version == model_version)
        .order_by(Evaluation.created_at.desc()).limit(min(max(limit, 1), 100)))).scalars().all()
    return {"model_version": model_version, "evals": [
        {"eval_id": str(r.eval_id), "verdict": r.verdict, "aggregate": r.aggregate,
         "challenger_of": r.challenger_of, "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows]}
