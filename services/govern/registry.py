"""Model registry (M4.4): the champion and challengers per task, each carrying its gold metrics (including
Safe-mIoU) and promotion history. Backed by model_registry; references a ModelRun by version. Reads are
the source of truth for which model serves and what it scored on the frozen gold set."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ModelRegistry, ModelRun

log = get_logger("govern_registry")


def _arch_of(base_weights: str | None) -> str | None:
    """The architecture family a checkpoint name denotes, or None when the name does not say.

    Read off the base weights rather than guessed from the run, because that string is the one thing that
    is always literally true about what was loaded. None rather than a default: "we do not know which
    architecture this is" and "it is a yolo11n" are different facts and the column has to be able to hold
    the first one.
    """
    if not base_weights:
        return None
    stem = str(base_weights).rsplit("/", 1)[-1]
    for suffix in (".pt", ".onnx", ".engine"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or None


async def register(db: AsyncSession, model_version: str, task: str, gold_metrics: dict,
                   dataset_commit: str | None = None, weights_uri: str | None = None,
                   notes: str | None = None, arch: str | None = None,
                   parent_version: str | None = None, teacher_version: str | None = None,
                   origin: str = "trained") -> dict:
    """Register a challenger (not champion yet). Idempotent on model_version.

    The lineage arguments (0110) say where the weights came from: `parent_version` is the checkpoint this
    one continued, `teacher_version` the model a distilled student was fit against, `origin` how it came to
    exist. They are set on insert and on update, so a row registered before lineage existed gains it the
    next time its run is registered rather than staying blank forever.
    """
    existing = await db.get(ModelRegistry, model_version)
    if existing is None:
        db.add(ModelRegistry(model_version=model_version, task=task, gold_metrics=gold_metrics,
                             is_champion=False, dataset_commit=dataset_commit, weights_uri=weights_uri,
                             notes=notes, arch=arch, parent_version=parent_version,
                             teacher_version=teacher_version, origin=origin))
    else:
        existing.gold_metrics = gold_metrics
        existing.task = task
        # Only fill what is missing: a later registration of the same version must not blank a lineage
        # that an earlier, better-informed caller already recorded.
        existing.arch = existing.arch or arch
        existing.parent_version = existing.parent_version or parent_version
        existing.teacher_version = existing.teacher_version or teacher_version
        if origin != "trained":
            existing.origin = origin
    await db.commit()
    log.info("registry.registered", model_version=model_version, task=task, origin=origin)
    return {"model_version": model_version, "task": task, "is_champion": False, "origin": origin}


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
                          dataset_commit=run.dataset_name, weights_uri=run.weights_uri, notes=run.notes,
                          arch=_arch_of(run.base_weights), parent_version=run.base_weights,
                          origin="trained")


async def list_models(db: AsyncSession, task: str | None = None) -> list[dict]:
    q = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    if task:
        q = q.where(ModelRegistry.task == task)
    rows = (await db.execute(q.limit(100))).scalars().all()
    return [{"model_version": r.model_version, "task": r.task, "is_champion": r.is_champion,
             "promoted_from": r.promoted_from, "gold_metrics": r.gold_metrics,
             "dataset_commit": r.dataset_commit,
             "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]
