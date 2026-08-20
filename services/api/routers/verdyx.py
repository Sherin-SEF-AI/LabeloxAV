"""VERDYX slice-evaluation endpoints: record a per-slice evaluation with its champion-challenger verdict, and
read the slice matrix. The metrics come from the existing services/training/eval and Safe-mIoU; the champion
gate stays in services/govern/champion and consumes these verdicts. Mounted under /api; the registry groups
/training, /govern, /models under VERDYX."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Evaluation
from services.api.deps import db_session, require_role
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


# Blind audits: the capture-recapture path that measures recall against a denominator the model did not
# build. Seeding and scoring sit behind a reviewer floor because an audit is a measurement instrument and
# a seeded one commits annotation budget; reading is open to anyone who can read an evaluation.


class SeedAuditIn(BaseModel):
    run_id: str
    n_frames: int = 200
    stratify_by: str = "density"
    score_thr: float = 0.25
    iou_thr: float = 0.5
    project_id: str | None = None
    notes: str | None = None


@router.post("/verdyx/blind-audit/seed")
async def blind_audit_seed(payload: SeedAuditIn, db: AsyncSession = Depends(db_session),
                           _user=Depends(require_role("reviewer"))):
    """Choose the frames and create the job that serves them blind. Nothing is measured yet."""
    from services.verdyx.blind_audit import seed_audit

    res = await seed_audit(db, run_id=payload.run_id, n_frames=payload.n_frames,
                           stratify_by=payload.stratify_by, score_thr=payload.score_thr,
                           iou_thr=payload.iou_thr, project_id=payload.project_id, notes=payload.notes)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/verdyx/blind-audit/{audit_id}/score")
async def blind_audit_score(audit_id: str, min_frames: int = 1,
                            db: AsyncSession = Depends(db_session),
                            _user=Depends(require_role("reviewer"))):
    """Match the two observations and persist the estimate. Refuses while too little of it is labelled."""
    from services.verdyx.blind_audit import score_audit

    res = await score_audit(db, audit_id, min_frames=min_frames)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/verdyx/blind-audit/{audit_id}/mark-labeled")
async def blind_audit_mark_labeled(audit_id: str, db: AsyncSession = Depends(db_session),
                                   _user=Depends(require_role("annotator"))):
    """Declare the auditor finished with every frame that is not already marked.

    Explicit rather than inferred, because a frame with no boxes on it is either an empty frame or one
    nobody opened, and scoring the second as the first reports the model as having missed nothing exactly
    where it was never checked.
    """
    from services.verdyx.blind_audit import mark_frames_labeled

    n = await mark_frames_labeled(db, UUID(audit_id))
    return {"audit_id": audit_id, "marked": n}


@router.get("/verdyx/blind-audits")
async def blind_audit_list(run_id: str | None = None, limit: int = 50,
                           db: AsyncSession = Depends(db_session)):
    from services.verdyx.blind_audit import list_audits

    return {"audits": await list_audits(db, run_id=run_id, limit=limit)}


@router.get("/verdyx/blind-audit/{audit_id}")
async def blind_audit_get(audit_id: str, db: AsyncSession = Depends(db_session)):
    from services.verdyx.blind_audit import audit_progress

    res = await audit_progress(db, audit_id)
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.get("/verdyx/blind-audit/{audit_id}/estimate")
async def blind_audit_estimate(audit_id: str, db: AsyncSession = Depends(db_session)):
    """Every persisted slice of the estimate: per stratum, per class, and pooled.

    An unmeasurable slice is returned as a row with measured false and a reason, not omitted. "We checked
    and cannot tell" is a different statement from "we did not check", and only one of them is safe to
    read as the absence of a problem.
    """
    from db.models import BlindAudit, RecaptureEstimateRow

    audit = await db.get(BlindAudit, UUID(audit_id))
    if audit is None:
        raise HTTPException(404, "audit not found")
    rows = (await db.execute(
        select(RecaptureEstimateRow).where(RecaptureEstimateRow.audit_id == audit.audit_id)
        .order_by(RecaptureEstimateRow.class_id.nulls_first(),
                  RecaptureEstimateRow.stratum.nulls_first()))).scalars().all()
    onto = None
    if any(r.class_id is not None for r in rows):
        from services.autolabel.ontology import get_ontology

        onto = get_ontology()
    return {
        "audit_id": audit_id, "run_id": str(audit.run_id), "gold_id": audit.gold_id,
        "status": audit.status, "estimator": rows[0].estimator if rows else None,
        "caveat": ("captures correlate positively on hard objects, so the estimated population is biased "
                   "down: these are a lower bound on what was missed and an upper bound on recall"),
        "slices": [{
            "stratum": r.stratum, "class_id": r.class_id,
            "class_name": onto.by_id(r.class_id).name if (onto and r.class_id is not None) else None,
            "measured": r.measured, "reason": r.reason,
            "population": r.population, "lo": r.population_lo, "hi": r.population_hi,
            "model_recall": r.model_recall, "recall_lo": r.recall_lo, "recall_hi": r.recall_hi,
            "human_recall": r.human_recall, "gold_recall": r.gold_recall,
            "overstatement": (round(r.gold_recall - r.model_recall, 6)
                              if (r.gold_recall is not None and r.model_recall is not None) else None),
            "n_both": r.n_both, "n_model_only": r.n_model_only, "n_human_only": r.n_human_only,
        } for r in rows],
    }


# Fitted auto-accept thresholds (ORACLYX). Fitting is a measurement; activating changes what the engine
# accepts without a human looking, so it is a separate call behind an admin floor.


class FitThresholdsIn(BaseModel):
    run_id: str
    min_support: int | None = None
    n_boot: int | None = None
    activate: bool = False


@router.post("/oraclyx/thresholds/fit")
async def thresholds_fit(payload: FitThresholdsIn, db: AsyncSession = Depends(db_session),
                         _user=Depends(require_role("reviewer"))):
    """Fit a per-class operating point from a run's recorded outcomes. Stored inactive unless asked."""
    from core.accel.np_threshold import MIN_SUPPORT, N_BOOTSTRAP
    from services.oraclyx.threshold_fit import fit_thresholds

    res = await fit_thresholds(db, run_id=payload.run_id,
                               min_support=payload.min_support or MIN_SUPPORT,
                               n_boot=payload.n_boot or N_BOOTSTRAP, activate=payload.activate)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/oraclyx/thresholds/{fit_id}/activate")
async def thresholds_activate(fit_id: str, db: AsyncSession = Depends(db_session),
                              _user=Depends(require_role("admin"))):
    """Put a fit into force, retiring whichever fit for the same model was in force before.

    Admin rather than reviewer: this changes what the engine will accept into the corpus without anybody
    looking at it, across every class at once.
    """
    from services.oraclyx.threshold_fit import activate_fit

    res = await activate_fit(db, fit_id)
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.get("/oraclyx/thresholds")
async def thresholds_active(model_version: str, db: AsyncSession = Depends(db_session)):
    """What is actually in force for a model, and how much of the ontology it covers.

    A class absent from `by_class` is falling back to the configured constant. That is the normal case and
    it is reported rather than hidden, because a constant nobody measured is not a precision floor.
    """
    from services.oraclyx.threshold_fit import active_thresholds

    act = await active_thresholds(db, model_version)
    return {"model_version": model_version, **act,
            "note": ("classes absent from by_class fall back to the configured constant, which was never "
                     "measured at the precision it claims")}


@router.get("/oraclyx/thresholds/{fit_id}")
async def thresholds_detail(fit_id: str, db: AsyncSession = Depends(db_session)):
    from db.models import ThresholdFit

    rows = (await db.execute(select(ThresholdFit).where(ThresholdFit.fit_id == UUID(fit_id))
                             .order_by(ThresholdFit.class_name))).scalars().all()
    if not rows:
        raise HTTPException(404, "fit not found")
    return {
        "fit_id": fit_id, "model_version": rows[0].model_version, "run_id": str(rows[0].run_id),
        "gold_id": rows[0].gold_id, "score_field": rows[0].score_field,
        "active": bool(rows[0].active), "n_classes": len(rows),
        "n_fitted": sum(1 for r in rows if r.measured),
        "classes": [{
            "class_id": r.class_id, "class_name": r.class_name, "alpha": r.alpha,
            "measured": r.measured, "reason": r.reason,
            "threshold": r.threshold, "lo": r.threshold_lo, "hi": r.threshold_hi,
            "config_threshold": r.config_threshold,
            "delta": (round(r.threshold - r.config_threshold, 4)
                      if r.threshold is not None and r.config_threshold is not None else None),
            "far_at": r.far_at, "accept_rate": r.accept_rate, "n_pairs": r.n_pairs,
            "n_positive": r.n_positive, "n_accept": r.n_accept, "n_boot_fit": r.n_boot_fit,
        } for r in rows],
    }


@router.get("/sanyx/compat-matrix")
async def compat_matrix_view(top: int = 25, db: AsyncSession = Depends(db_session)):
    """The learned class-compatibility matrix, with the support each cell rests on.

    Cells are ordered by support rather than by probability: a cell at 1.0 on one observation is not a
    finding, and ordering by probability would put exactly those at the top.
    """
    from services.autolabel.compat_matrix import matrix_report

    return await matrix_report(db, top=top)


# Confusion-clique active learning (SIEVYX): which frames to label next, and how the budget is split.


@router.post("/sievyx/cliques/select")
async def cliques_select(run_id: str, budget: int = 200, seed: int | None = None,
                         db: AsyncSession = Depends(db_session),
                         _user=Depends(require_role("reviewer"))):
    """Select frames to label next, split across confusion cliques and ranked within each."""
    from services.sievyx.clique_sampler import _SEED, select_frames

    res = await select_frames(db, run_id=run_id, budget=budget, seed=seed or _SEED)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


class CliqueRewardIn(BaseModel):
    clique: str
    allocated: int
    recall_before: float | None = None
    recall_after: float | None = None


@router.post("/sievyx/cliques/reward")
async def cliques_reward(payload: CliqueRewardIn, db: AsyncSession = Depends(db_session),
                         _user=Depends(require_role("reviewer"))):
    """Report what a labelled batch bought, so the posterior for that clique can move."""
    from services.sievyx.clique_sampler import record_reward

    return await record_reward(db, clique=payload.clique, allocated=payload.allocated,
                               recall_before=payload.recall_before, recall_after=payload.recall_after)


@router.get("/sievyx/cliques")
async def cliques_report(db: AsyncSession = Depends(db_session)):
    """The posteriors, and whether any of them has ever been moved by a labelling cycle."""
    from services.sievyx.clique_sampler import bandit_report

    return await bandit_report(db)


@router.get("/verdyx/hierarchical")
async def hierarchical_eval(run_id: str, gold_id: str | None = None, score_thr: float = 0.0,
                            db: AsyncSession = Depends(db_session)):
    """AP at every level of the class tree, and what the gap between levels says to work on.

    A large leaf-to-l1 gap means the detector finds objects and names them wrong, so labelling class
    boundaries buys AP. A small one means it is not finding them, and more class labels will not help.
    """
    from services.verdyx.hier_eval import evaluate_hierarchical

    res = await evaluate_hierarchical(db, run_id=run_id, gold_id=gold_id, score_thr=score_thr)
    if not res.get("measured"):
        raise HTTPException(400, res.get("reason", "not measurable"))
    return res
