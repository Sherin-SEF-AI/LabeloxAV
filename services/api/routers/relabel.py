"""Relabel endpoints (M4.2): start a relabel run (ontology promotion locally, model re-inference on the
A100 burst), list runs, ingest cloud-pod output, and revert a run."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import RelabelRun
from services.api.deps import db_session
from services.relabel.apply import revert_run
from services.relabel.run import ingest_model_relabel, start_relabel

router = APIRouter()


class OntologyPromotion(BaseModel):
    from_class: str
    to_class: str


class RelabelStartIn(BaseModel):
    model_version: str
    session_ids: list[str] | None = None
    ontology_promotion: OntologyPromotion | None = None
    compute_target: str = "local"


@router.post("/relabel/start")
async def start(payload: RelabelStartIn):
    return await start_relabel(payload.model_version, payload.session_ids,
                               payload.ontology_promotion.model_dump() if payload.ontology_promotion else None,
                               payload.compute_target)


@router.get("/relabel/runs")
async def runs(db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(RelabelRun).order_by(RelabelRun.created_at.desc()).limit(50))).scalars().all()
    return [{"run_id": str(r.run_id), "model_version": r.model_version, "lakefs_branch": r.lakefs_branch,
             "proposed": r.proposed, "auto_applied": r.auto_applied, "routed_to_review": r.routed_to_review,
             "regressions_flagged": r.regressions_flagged, "reason": r.reason,
             "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]


class IngestIn(BaseModel):
    model_version: str
    model_output: list[dict]
    session_ids: list[str] | None = None


@router.post("/relabel/ingest")
async def ingest(payload: IngestIn):
    return await ingest_model_relabel(payload.model_version, payload.model_output, payload.session_ids)


@router.post("/relabel/runs/{run_id}/revert")
async def revert(run_id: str, db: AsyncSession = Depends(db_session)):
    return await revert_run(db, run_id)
