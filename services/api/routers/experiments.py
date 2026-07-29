"""Experiments, tracker scoring, and false-negative mining: the three closed-loop surfaces that were absent.

Grouped in one router because they share an audience. Each answers a question the loop could ask and could
not get an answer to: is this line of training work improving, how good is the tracker, and what is the
model missing entirely.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role, require_user

router = APIRouter()


# ---------------------------------------------------------------- experiments

class ExperimentIn(BaseModel):
    name: str
    task_type: str = "detection"
    description: str | None = None
    hypothesis: str | None = None
    tags: list[str] = []


@router.get("/experiments")
async def list_experiments(task_type: str | None = None, db: AsyncSession = Depends(db_session)):
    from services.training.experiments import list_experiments as _list

    return await _list(db, task_type=task_type)


@router.post("/experiments", dependencies=[Depends(require_role("reviewer"))])
async def create_experiment(payload: ExperimentIn, user=Depends(require_user),
                            db: AsyncSession = Depends(db_session)):
    from services.training.experiments import create_experiment as _create

    return await _create(db, name=payload.name, task_type=payload.task_type,
                         description=payload.description, hypothesis=payload.hypothesis,
                         tags=payload.tags, created_by=getattr(user, "name", None))


@router.get("/experiments/{name}")
async def experiment_detail(name: str, metric: str = "map50",
                            db: AsyncSession = Depends(db_session)):
    from services.training.experiments import experiment_detail as _detail

    try:
        return await _detail(db, name, metric=metric)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class AttachIn(BaseModel):
    job_id: str
    label: str | None = None
    baseline_run_id: str | None = None
    notes: str | None = None


@router.post("/experiments/{name}/runs", dependencies=[Depends(require_role("reviewer"))])
async def attach_run(name: str, payload: AttachIn, db: AsyncSession = Depends(db_session)):
    from services.training.experiments import attach_run as _attach

    try:
        return await _attach(db, experiment=name, job_id=payload.job_id, label=payload.label,
                             baseline_run_id=payload.baseline_run_id, notes=payload.notes)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/experiments/runs/compare")
async def compare(a: str, b: str, db: AsyncSession = Depends(db_session)):
    """Two runs side by side: what differed going in, and what it did coming out."""
    from services.training.experiments import compare_runs

    try:
        return await compare_runs(db, a, b)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- tracker scoring

@router.post("/eval/tracking", dependencies=[Depends(require_role("reviewer"))])
async def score_tracking(gold_id: str, run_id: str, iou_thr: float = Query(0.5, ge=0.1, le=0.95),
                         db: AsyncSession = Depends(db_session)):
    """MOTA, IDF1 and HOTA for one inference run against one sealed gold set.

    422 rather than 500 when the gold set carries no identities: it is a well-formed request against a set
    that cannot answer it, and the message says which objects are missing a track.
    """
    from services.verdyx.track_eval import TrackGroundTruthUnavailable, score_tracker

    try:
        return await score_tracker(db, gold_id=gold_id, run_id=run_id, iou_thr=iou_thr)
    except TrackGroundTruthUnavailable as exc:
        raise HTTPException(422, str(exc)) from exc


# ---------------------------------------------------------------- false negatives

@router.get("/activelearn/false-negatives")
async def false_negatives(session_id: str | None = None,
                          limit: int = Query(4000, ge=1, le=20000),
                          top_k: int = Query(200, ge=1, le=2000),
                          accept_threshold: float = Query(0.5, ge=0.0, le=1.0),
                          db: AsyncSession = Depends(db_session)):
    """Frames the detector probably missed something in.

    Frames rather than objects, because the reviewer's task here is to draw what is absent, which is a
    different action from correcting a box and does not belong in the same queue.
    """
    from services.activelearn.false_negatives import mine_false_negatives

    return await mine_false_negatives(db, session_id=session_id, limit=limit, top_k=top_k,
                                      accept_threshold=accept_threshold)


@router.post("/activelearn/recall-reliability", dependencies=[Depends(require_role("reviewer"))])
async def recall_reliability(apply: bool = True, min_verdicts: int = Query(20, ge=1, le=1000),
                             db: AsyncSession = Depends(db_session)):
    """Fit the per-channel confirmed rates from the human verdicts, replacing the hand-set priors.

    The recall channels have been ranked by three numbers somebody guessed while the verdicts needed to
    measure them accumulated in the same table.
    """
    from services.recall.recover import fit_channel_reliability

    return await fit_channel_reliability(db, min_verdicts=min_verdicts, apply=apply)
