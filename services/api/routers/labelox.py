"""LABELOX core spine endpoints. The three-path auto-label, workspace, ontology, active learning, and recall
recovery keep their existing routers (autolabel, objects, review, triage, multicam, recall); this adds the M4
integration point: the gate-integrated label queue that subtracts the SANYX/CALYX gates from the SIEVYX
priority, so the workspace only ever offers samples that are both worth labeling and safe to label."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.labelox.learned_reconcile import reconcile_parity
from services.labelox.propagate4d import propagate_track
from services.labelox.quality import gold_audit, inter_annotator_agreement
from services.labelox.queue import build_label_queue

router = APIRouter()


class ParityIn(BaseModel):
    learned_metrics: dict
    heuristic_metrics: dict
    margin: float | None = None


@router.post("/labelox/reconcile/parity")
async def reconcile_parity_gate(payload: ParityIn):
    """The learned-reconciler parity gate: may the learned fusion head replace the heuristic on this held-out?"""
    return reconcile_parity(payload.learned_metrics, payload.heuristic_metrics, payload.margin)


class Propagate4dIn(BaseModel):
    keyframe_box: list[float]
    n_frames: int
    velocity: list[float] | None = None
    known: dict[int, list[float]] | None = None


@router.post("/labelox/propagate4d")
async def propagate4d(payload: Propagate4dIn):
    """Propagate an annotation across a clip with a single stable track identity, healing gaps by interpolation."""
    return propagate_track(payload.keyframe_box, payload.n_frames, payload.velocity, payload.known)


class AuditIn(BaseModel):
    predicted: list[dict]
    gold: list[dict]
    iou_thr: float = 0.5


@router.post("/labelox/quality/audit")
async def quality_audit(payload: AuditIn):
    """Gold-set audit: match annotations to gold and flag the ones that fail (catches bad labels)."""
    return gold_audit(payload.predicted, payload.gold, payload.iou_thr)


class AgreementIn(BaseModel):
    annotations: list[dict]
    iou_thr: float = 0.5


@router.post("/labelox/quality/agreement")
async def quality_agreement(payload: AgreementIn):
    """Inter-annotator agreement for the same object across annotators."""
    return {"agreement": inter_annotator_agreement(payload.annotations, payload.iou_thr)}


@router.get("/labelox/queue")
async def label_queue(limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The gated, SIEVYX-ranked label queue. Samples from SANYX-quarantined or CALYX-blocked sessions are
    excluded; the rest are ordered by the combined priority score."""
    return await build_label_queue(db, limit)
