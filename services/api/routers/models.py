"""Model-run endpoints: the eval/promotion ledger for the close-the-loop fine-tunes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ModelRun
from services.api.deps import db_session

router = APIRouter()


@router.get("/models")
async def list_models(db: AsyncSession = Depends(db_session), limit: int = Query(200, ge=1, le=1000)):
    rows = (await db.execute(select(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit))).scalars().all()
    return [
        {
            "run_id": m.run_id,
            "base_weights": m.base_weights,
            "dataset": m.dataset_name,
            "n_train": m.n_train,
            "n_val": m.n_val,
            "epochs": m.epochs,
            "baseline_map50": m.baseline_metrics.get("map50"),
            "candidate_map50": m.metrics.get("map50"),
            "map50_delta": m.gate.get("map50_delta"),
            "promote": m.gate.get("promote"),
            "promoted": m.promoted,
            "reasons": m.gate.get("reasons"),
            "ontology_version": m.ontology_version,
        }
        for m in rows
    ]
