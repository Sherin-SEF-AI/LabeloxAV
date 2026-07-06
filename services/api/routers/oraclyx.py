"""ORACLYX offline-fusion pseudo-GT endpoints: record a consensus verdict for an object, read the
consensus/disagreement board, and export the distillation set. The multi-view/LiDAR fusion itself lives in
the existing services/lidar and services/autolabel/fusion; this router adds the consensus gate and the
distillation export. Mounted under /api; the platform registry groups /lidar and /hdmap under ORACLYX."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.oraclyx.run import consensus_summary, export_distillation, record_consensus

router = APIRouter()


class Detection(BaseModel):
    path: str
    class_id: int
    bbox: list[float]
    conf: float = 0.0


class ConsensusIn(BaseModel):
    fusion: Detection
    auto_paths: list[Detection]
    fusion_run_id: str | None = None


@router.post("/oraclyx/object/{object_id}/consensus")
async def consensus(object_id: uuid.UUID, payload: ConsensusIn):
    """Vote fusion against the three auto-label paths for an object, persist the pseudo-label, and route the
    object (auto_accept on consensus, review on disagreement)."""
    return await record_consensus(object_id, payload.fusion.model_dump(),
                                  [p.model_dump() for p in payload.auto_paths], payload.fusion_run_id)


@router.get("/oraclyx/board")
async def board(db: AsyncSession = Depends(db_session)):
    """The consensus/disagreement board: auto-accepted vs human-routed pseudo-label counts."""
    return await consensus_summary(db)


@router.get("/oraclyx/distillation")
async def distillation(min_score: float = 0.5, db: AsyncSession = Depends(db_session)):
    """The distillation manifest: consensus pseudo-labels with soft targets, the edge-model training signal."""
    return await export_distillation(db, min_score=min_score)
