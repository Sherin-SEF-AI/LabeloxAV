"""VERDYX slice-evaluation endpoints: record a per-slice evaluation with its champion-challenger verdict, and
read the slice matrix. The metrics come from the existing services/training/eval and Safe-mIoU; the champion
gate stays in services/govern/champion and consumes these verdicts. Mounted under /api; the registry groups
/training, /govern, /models under VERDYX."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Evaluation
from services.api.deps import db_session
from services.verdyx.run import matrix, record_evaluation
from services.verdyx.safety_recall import (
    critical_object_recall,
    near_miss_slice,
    ttc_weighted_recall,
)
from services.verdyx.shadow import regression_triage
from services.verdyx.stats import bootstrap_ci, paired_significance

router = APIRouter()


class SafetyIn(BaseModel):
    objects: list[dict]              # [{object_id, class_id, detected, ttc_s}]
    ttc_horizon_s: float = 6.0
    near_miss_ttc_s: float = 2.0


@router.post("/verdyx/safety/recall")
async def safety_recall(payload: SafetyIn):
    """The safety-weighted recall panel: critical-object recall, TTC-weighted recall, and the near-miss slice
    with the missed-object escalation list."""
    return {"critical": critical_object_recall(payload.objects),
            "ttc_weighted": ttc_weighted_recall(payload.objects, payload.ttc_horizon_s),
            "near_miss": near_miss_slice(payload.objects, payload.near_miss_ttc_s)}


class BootstrapIn(BaseModel):
    values: list[float]
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 12345


@router.post("/verdyx/stats/bootstrap")
async def stats_bootstrap(payload: BootstrapIn):
    """A percentile bootstrap CI on a per-object metric, so a point estimate carries its uncertainty."""
    return bootstrap_ci(payload.values, payload.n_boot, payload.alpha, payload.seed)


class SignificanceIn(BaseModel):
    champion: list[float]
    challenger: list[float]
    n_perm: int = 2000
    seed: int = 12345


@router.post("/verdyx/stats/significance")
async def stats_significance(payload: SignificanceIn):
    """A paired permutation test: is the challenger's per-object gain over the champion real or chance?"""
    return paired_significance(payload.champion, payload.challenger, payload.n_perm, payload.seed)


class TriageIn(BaseModel):
    baseline: dict[str, float]
    current: dict[str, float]
    margin: float = 0.05
    protected: list[str] | None = None


@router.post("/verdyx/shadow/triage")
async def shadow_triage(payload: TriageIn):
    """Regression triage over a shadow snapshot: name the regressed slices worst-first and alarm on a protected
    slice drop."""
    return regression_triage(payload.baseline, payload.current, payload.margin,
                             set(payload.protected or []))


class EvalIn(BaseModel):
    model_version: str
    aggregate: dict                 # {map50, map, precision, recall, safe_miou}
    # Per-slice metrics are COMPUTED from the prediction plane when run_id is given (the trustworthy path).
    # Supplying them directly is retained only for backfilling a historical evaluation, and is recorded as
    # caller-supplied so a hand-typed number is never mistaken for a measured one.
    per_slice: dict | None = None
    run_id: str | None = None       # the InferenceRun to score; with gold_id this computes per_slice
    challenger_of: str | None = None
    release_commit: str | None = None
    gold_id: str | None = None
    protected: list[str] | None = None


@router.post("/verdyx/evaluate")
async def evaluate(payload: EvalIn, db: AsyncSession = Depends(db_session)):
    """Record a per-slice evaluation and compute the protected-slice verdict governance consumes.

    Given run_id + gold_id the slice metrics are computed here from the immutable prediction plane, so the
    safety gate reads numbers derived from the model it is judging. Previously per_slice arrived in the
    request body and nothing in the tree produced it, which made the protected-slice gate only as trustworthy
    as the JSON someone typed.
    """
    per_slice = payload.per_slice or {}
    source = "caller_supplied"
    if payload.run_id and payload.gold_id:
        from services.verdyx.slice_eval import compute_slice_metrics

        computed = await compute_slice_metrics(db, payload.gold_id, run_id=payload.run_id,
                                               slice_ids=payload.protected)
        if "error" in computed:
            raise HTTPException(status_code=400, detail=computed["error"])
        per_slice, source = computed, "computed_from_prediction_plane"
    elif not per_slice:
        raise HTTPException(
            status_code=400,
            detail="supply run_id + gold_id to compute per-slice metrics, or per_slice to backfill one")

    result = await record_evaluation(db, payload.model_version, payload.aggregate, per_slice,
                                     challenger_of=payload.challenger_of,
                                     release_commit=payload.release_commit,
                                     gold_id=payload.gold_id, protected=payload.protected)
    result["per_slice_source"] = source
    return result


@router.get("/verdyx/pairs")
async def pairs(db: AsyncSession = Depends(db_session)):
    """Recent challenger/champion pairs that have evaluations, so the matrix has something to compare."""
    rows = (await db.execute(
        select(Evaluation.model_version, Evaluation.challenger_of, Evaluation.verdict, Evaluation.created_at)
        .where(Evaluation.challenger_of.isnot(None)).order_by(Evaluation.created_at.desc()).limit(20))).all()
    seen = set()
    out = []
    for mv, champ, verdict, _ in rows:
        key = (champ, mv)
        if key in seen:
            continue
        seen.add(key)
        out.append({"champion": champ, "challenger": mv, "verdict": verdict})
    return {"pairs": out}


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
