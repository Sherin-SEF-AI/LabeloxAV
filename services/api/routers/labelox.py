"""LABELOX core spine endpoints. The three-path auto-label, workspace, ontology, active learning, and recall
recovery keep their existing routers (autolabel, objects, review, triage, multicam, recall); this adds the M4
integration point: the gate-integrated label queue that subtracts the SANYX/CALYX gates from the SIEVYX
priority, so the workspace only ever offers samples that are both worth labeling and safe to label."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.labelox.queue import build_label_queue

router = APIRouter()


@router.get("/labelox/queue")
async def label_queue(limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The gated, SIEVYX-ranked label queue. Samples from SANYX-quarantined or CALYX-blocked sessions are
    excluded; the rest are ordered by the combined priority score."""
    return await build_label_queue(db, limit)
