"""Model registry (M4.4): the champion and challengers per task, each carrying its gold metrics (including
Safe-mIoU) and promotion history. Backed by model_registry; references a ModelRun by version. Reads are
the source of truth for which model serves and what it scored on the frozen gold set."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ModelRegistry, ModelRun

log = get_logger("govern_registry")


async def register(db: AsyncSession, model_version: str, task: str, gold_metrics: dict,
                   dataset_commit: str | None = None, weights_uri: str | None = None,
                   notes: str | None = None) -> dict:
    """Register a challenger (not champion yet). Idempotent on model_version."""
    existing = await db.get(ModelRegistry, model_version)
    if existing is None:
        db.add(ModelRegistry(model_version=model_version, task=task, gold_metrics=gold_metrics,
                             is_champion=False, dataset_commit=dataset_commit, weights_uri=weights_uri, notes=notes))
    else:
        existing.gold_metrics = gold_metrics
        existing.task = task
    await db.commit()
    log.info("registry.registered", model_version=model_version, task=task)
    return {"model_version": model_version, "task": task, "is_champion": False}


async def get_champion(db: AsyncSession, task: str) -> ModelRegistry | None:
    return (await db.execute(
        select(ModelRegistry).where(ModelRegistry.task == task, ModelRegistry.is_champion.is_(True)))).scalars().first()


async def set_champion(db: AsyncSession, model_version: str, task: str, promoted_from: str | None) -> None:
    """Make model_version the sole champion for the task (demote any incumbent)."""
    await db.execute(update(ModelRegistry).where(ModelRegistry.task == task).values(is_champion=False))
    reg = await db.get(ModelRegistry, model_version)
    if reg is not None:
        reg.is_champion = True
        reg.promoted_from = promoted_from
    await db.commit()


async def register_from_run(db: AsyncSession, run_id: str, task: str | None = None) -> dict:
    """Pull a ModelRun's metrics into the registry as a challenger."""
    run = await db.get(ModelRun, run_id)
    if run is None:
        return {"error": "model run not found"}
    return await register(db, run_id, task or run.task_type, run.metrics or {},
                          dataset_commit=run.dataset_name, weights_uri=run.weights_uri, notes=run.notes)


async def list_models(db: AsyncSession, task: str | None = None) -> list[dict]:
    q = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    if task:
        q = q.where(ModelRegistry.task == task)
    rows = (await db.execute(q.limit(100))).scalars().all()
    return [{"model_version": r.model_version, "task": r.task, "is_champion": r.is_champion,
             "promoted_from": r.promoted_from, "gold_metrics": r.gold_metrics,
             "dataset_commit": r.dataset_commit,
             "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]
