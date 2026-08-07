"""Evaluate registered models on a COMMON sealed gold set (C5).

The promotion gate used to compare each model's stored gold_metrics, but those were produced by each training
run against its OWN val split (services/training/jobs.py evaluates the freshly-trained weights on the dataset
build it just made). Comparing a challenger scored on split A against a champion scored on split B is
apples-to-oranges: a challenger can "beat" the champion purely because its split was easier. This module
re-scores both models on the SAME frozen gold set, aligned to each model's own class order, so the numbers the
gate compares are commensurable.

Heavy (GPU + model load), so callers run it off the event loop; it degrades gracefully to None when the model
weights or a gold set are unavailable (e.g. unit tests), letting the caller fall back to stored metrics with
an explicit, audited caveat rather than crashing.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import GoldSet, ModelRegistry

log = get_logger("govern_gold_eval")


async def latest_gold_id(db: AsyncSession, ontology_version: str | None = None) -> str | None:
    """The most recently sealed gold set (optionally pinned to an ontology version). This is the common
    yardstick both models are scored against."""
    q = select(GoldSet).order_by(GoldSet.created_at.desc())
    if ontology_version:
        q = q.where(GoldSet.ontology_version == ontology_version)
    g = (await db.execute(q.limit(1))).scalars().first()
    return g.gold_id if g else None


def _load_weights_and_names(weights_uri: str, local_path: str) -> list:
    """Download the model weights (once) and read its class order. Pure/blocking: runs in a worker thread."""
    from pathlib import Path

    from ultralytics import YOLO

    from core.storage import get_object_store

    if not Path(local_path).exists():
        Path(local_path).write_bytes(get_object_store().get_bytes(weights_uri))
    names = YOLO(local_path).names
    return [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)


def _score_yaml(local_weights: str, data_yaml: str, gold_id: str, imgsz: int) -> dict:
    """Detection + Safe-mIoU eval on an already-materialized gold split. Blocking GPU work (worker thread)."""
    from services.training.eval import evaluate, safe_miou_report

    metrics = evaluate(local_weights, data_yaml, "val", imgsz)
    try:
        sm = safe_miou_report(local_weights, data_yaml, "val", imgsz)
        if sm.get("safe_miou") is not None:
            metrics["safe_miou"] = sm["safe_miou"]
    except Exception as exc:  # noqa: BLE001
        log.warning("gold_eval.safe_miou_failed", gold_id=gold_id, error=str(exc))
    metrics["gold_id"] = gold_id
    return metrics


async def evaluate_on_gold(db: AsyncSession, model_version: str, gold_id: str) -> dict | None:
    """Score one registered model on the sealed gold set, aligned to the model's class order. Returns metrics
    stamped with gold_id, or None when the model has no downloadable weights (nothing to score)."""
    reg = await db.get(ModelRegistry, model_version)
    if reg is None or not reg.weights_uri:
        return None

    settings = get_settings()
    imgsz = getattr(settings.training, "eval_imgsz", 960)
    scratch = settings.scratch_path() / "gold_eval"
    scratch.mkdir(parents=True, exist_ok=True)
    local = str(scratch / f"{model_version}.pt")
    loop = asyncio.get_event_loop()
    try:
        # 1. fetch weights + read the model's class order (thread: network + torch load)
        names_list = await loop.run_in_executor(None, _load_weights_and_names, reg.weights_uri, local)
        # 2. align the frozen gold labels to that class order (async: DB read for the sealed object ids)
        from services.training.gold import materialize_for_model

        data_yaml = await materialize_for_model(gold_id, names_list)
        # 3. score (thread: GPU val pass)
        metrics = await loop.run_in_executor(None, _score_yaml, local, data_yaml, gold_id, imgsz)
        # 4. reconcile against the prediction plane: run inference once, score it, and compare map50 to ap50 on
        #    the SAME gold set + model. Two harnesses that disagree beyond epsilon is a measurement fault, so
        #    flag the metrics divergent (the champion gate refuses to promote on that flag). Degrades silently
        #    when inference is unavailable (no GPU / weights), keeping the val-pass result.
        await _reconcile_with_prediction_plane(db, model_version, gold_id, metrics,
                                              vocabulary=frozenset(names_list))
        return metrics
    except Exception as exc:  # noqa: BLE001
        log.warning("gold_eval.failed", model_version=model_version, gold_id=gold_id, error=str(exc))
        return None


async def _reconcile_with_prediction_plane(db: AsyncSession, model_version: str, gold_id: str,
                                           metrics: dict,
                                           vocabulary: frozenset[str] | None = None) -> None:
    from services.analytics.evaluation import evaluate_gold_patches
    from services.verdyx.inference_run import run_inference_on_gold

    try:
        run_id = await run_inference_on_gold(db, model_version, gold_id)
        if run_id is None:
            metrics["reconcile"] = "prediction plane unavailable (weights/GPU); val-pass only"
            return
        # The model's own class list, the same one the val pass aligned the gold labels to. Passing it makes
        # both harnesses score one population; without it they disagree by exactly the objects this model was
        # never taught, which is a measurement fault reported as a model fault.
        pp = await evaluate_gold_patches(db, gold_id, run_id=run_id, model_vocabulary=vocabulary)
        metrics["prediction_plane_scored"] = pp.get("gold_scored")
        metrics["prediction_plane_resolvable"] = pp.get("gold_resolvable")
        ap50 = pp.get("ap50")
        val_map50 = metrics.get("map50")
        if ap50 is None or val_map50 is None:
            metrics["reconcile"] = "one harness produced no comparable number"
            return
        eps = get_settings().phase4.govern.harness_reconcile_epsilon
        delta = abs(float(val_map50) - float(ap50))
        metrics["prediction_plane_ap50"] = ap50
        metrics["prediction_plane_run_id"] = run_id
        metrics["harness_delta"] = round(delta, 4)
        if delta > eps:
            metrics["harness_divergent"] = True
            log.error("gold_eval.harness_divergence", model_version=model_version, gold_id=gold_id,
                      val_map50=val_map50, prediction_plane_ap50=ap50, delta=round(delta, 4), epsilon=eps)
    except Exception as exc:  # noqa: BLE001 - a reconcile failure must not crash the promotion path
        log.warning("gold_eval.reconcile_failed", model_version=model_version, gold_id=gold_id, error=str(exc))
        metrics["reconcile"] = f"reconcile error: {type(exc).__name__}"
