"""VERDYX persistence: record a per-slice Evaluation with its champion-challenger verdict onto the evaluation
table (the first-class eval rows added in migration 0049), and read the slice matrix for the UI. The metrics
themselves are produced by the existing services/training/eval (per-slice mAP/precision/recall) and Safe-mIoU;
this wires them to the spine and the protected-slice gate."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Evaluation, GovernanceState
from services.verdyx.verdict import slice_matrix, slice_verdict

log = get_logger("verdyx.run")


def _protected(cfg) -> list[str]:
    # Config override wins; otherwise the active pack's eval_strata.protected_slices (AV:
    # ["pedestrian_night", "autorickshaw_glare"]). GovernSettings.protected_slices now exists (defaults empty),
    # so the pack default flows through instead of the old hardcoded literal.
    from services.domain import protected_slices
    return list(getattr(cfg, "protected_slices", []) or protected_slices())


async def record_evaluation(db: AsyncSession, model_version: str, aggregate: dict, per_slice: dict,
                            challenger_of: str | None = None, release_commit: str | None = None,
                            gold_id: str | None = None, protected: list[str] | None = None) -> dict:
    """Score a model per slice, compute the verdict against the current champion (or a named one), and persist
    an Evaluation row."""
    protected = protected or _protected(get_settings().phase4.govern if hasattr(get_settings(), "phase4") else None)
    champ_version = challenger_of
    if champ_version is None:
        state = await db.get(GovernanceState, 1)
        champ_version = state.champion_version if state else None

    verdict = {"verdict": "needs_review", "reasons": ["no champion to compare against"], "regressed_slices": []}
    if champ_version:
        champ_eval = (await db.execute(
            select(Evaluation).where(Evaluation.model_version == champ_version)
            .order_by(Evaluation.created_at.desc()).limit(1))).scalar_one_or_none()
        if champ_eval is not None:
            verdict = slice_verdict(
                {"aggregate": champ_eval.aggregate, "per_slice": champ_eval.per_slice},
                {"aggregate": aggregate, "per_slice": per_slice}, protected)

    row = Evaluation(model_version=model_version, release_commit=release_commit, gold_id=gold_id,
                     per_slice=per_slice, aggregate=aggregate, verdict=verdict["verdict"],
                     challenger_of=champ_version, failure_clusters={})
    db.add(row)
    await db.commit()

    # Group the misses now that the eval has an id, so failure-driven mining has an input. This was written
    # `{}` unconditionally, which left services/sievyx/failure_mining.py without a caller or a source.
    # Clustering is best-effort: a model whose failures cannot be grouped is still a model that evaluated,
    # and the verdict above must not depend on it.
    try:
        from services.verdyx.failure_clusters import build_failure_clusters, resolve_patch_eval_id

        # Resolved by (gold_id, model_version), not by eval id: EvalPatch.eval_id is minted by the analytics
        # evaluator and shares no id space with this table.
        patch_eval = await resolve_patch_eval_id(db, gold_id=gold_id, model_version=model_version)
        row.failure_clusters = await build_failure_clusters(db, patch_eval)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("verdyx.failure_clusters_failed", eval_id=str(row.eval_id), error=str(exc))
    log.info("verdyx.eval", model=model_version, verdict=verdict["verdict"],
             regressed=[r["slice"] for r in verdict.get("regressed_slices", [])])
    return {"eval_id": str(row.eval_id), "model_version": model_version, "verdict": verdict,
            "challenger_of": champ_version}


async def matrix(db: AsyncSession, champion_version: str, challenger_version: str,
                 slice_metric: str = "map") -> dict:
    """The slice matrix (champion vs challenger per slice) for the UI."""
    async def _latest(v):
        return (await db.execute(select(Evaluation).where(Evaluation.model_version == v)
                                 .order_by(Evaluation.created_at.desc()).limit(1))).scalar_one_or_none()
    cp, ch = await _latest(champion_version), await _latest(challenger_version)
    if cp is None or ch is None:
        return {"error": "missing evaluation for champion or challenger"}
    slices = sorted(set(cp.per_slice) | set(ch.per_slice))
    return {"champion": champion_version, "challenger": challenger_version,
            "rows": slice_matrix({"per_slice": cp.per_slice}, {"per_slice": ch.per_slice}, slices, slice_metric)}
