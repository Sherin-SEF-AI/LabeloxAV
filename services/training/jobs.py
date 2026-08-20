"""Training job spec, enqueue, and the shared executor. enqueue_job writes a pending training_job row;
run_job (called by the worker) drives any task plugin through build -> baseline eval -> train ->
candidate eval -> gate -> record model_run -> optional promote, streaming progress onto the job row.

Training is blocking, so it runs in the loop's executor thread; per-epoch progress is marshalled back
to the loop via an asyncio.Queue. This keeps the worker's single event loop + cached engine intact.
"""

from __future__ import annotations

import asyncio
import re
import threading
import uuid

from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging import get_logger
from core.observability import spawn
from core.storage import get_object_store
from db.models import ModelRun, TrainingJob
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.finetune import _run_id
from services.training.lineage import dataset_fingerprint, record_dataset_commit
from services.training.tasks import get_task

log = get_logger("training_jobs")

_SLIM_KEYS = ("map50", "map", "precision", "recall", "safe_miou")


class TrainJobSpec(BaseModel):
    purpose: str = "perception"            # the model line, e.g. "vru-detector", "blr-detector"
    task_type: str = "detection"
    compute_target: str = "local"          # local (RTX 5080) | cloud (RunPod A100); heavy jobs -> cloud
    dataset_spec: dict = Field(default_factory=dict)   # BuildSpec fields, or {"data_yaml": ...}
    base_weights: str | None = None        # None -> task.default_base_weights()
    hparams: dict = Field(default_factory=dict)        # epochs, imgsz, batch
    gate: dict = Field(default_factory=dict)           # min_map_delta, max_class_drop, min_safe_miou
    promote: bool = False
    notes: str | None = None


async def _promote_run(db, run_id: str, task_type: str) -> dict:
    """Actually promote, rather than recording that promotion was wanted.

    `promote: true` used to set a boolean on the run row and log a line telling somebody to export an
    environment variable by hand. Nothing became champion, nothing served the new weights, and the run said
    "promoted" while the old model kept answering every request. This routes through the same champion gate
    as POST /govern/promote, so the gold-set comparison, the auto-promote kill switch and the single-champion
    constraint all apply, and the same notification goes out when the gate refuses: a blocked promotion that
    nobody is told about is how the flywheel idles for a day.
    """
    from services.govern.champion import evaluate_and_promote
    from services.notify import notify

    try:
        result = await evaluate_and_promote(db, run_id, task_type)
    except Exception as exc:  # noqa: BLE001
        # A promotion failure must not discard a finished training run: the weights are already stored and
        # the run row is already written, and losing all of that to a governance error would be worse than
        # the missed promotion.
        log.warning("training.promote_failed", run_id=run_id, error=str(exc))
        return {"promoted": False, "error": str(exc)}

    promoted = bool(result.get("promoted"))
    try:
        await notify(
            db, kind="model_promoted" if promoted else "promotion_blocked",
            severity="info" if promoted else "warn",
            title=(f"{task_type} model {run_id} promoted" if promoted
                   else f"{task_type} promotion blocked: {run_id}"),
            body=result.get("reason") or result.get("detail"),
            href="/govern", subject_type="model", subject_id=run_id,
            meta={"task": task_type, "metrics": result.get("metrics") or {}})
    except Exception as exc:  # noqa: BLE001
        log.warning("training.promote_notify_failed", run_id=run_id, error=str(exc))
    log.info("training.promotion", run_id=run_id, promoted=promoted, reason=result.get("reason"))
    return result


def _safe_name(purpose: str, job_id) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", purpose).strip("-") or "model"
    return f"{base}-{str(job_id)[:8]}"


def _slim(metrics: dict) -> dict:
    return {k: metrics[k] for k in _SLIM_KEYS if k in metrics}


async def enqueue_job(spec: TrainJobSpec) -> str:
    job_id = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(TrainingJob(
            job_id=job_id, status="pending", purpose=spec.purpose, task_type=spec.task_type,
            compute_target=spec.compute_target, config=spec.model_dump(), progress=0.0,
        ))
        await db.commit()
    log.info("training.enqueued", job_id=str(job_id), purpose=spec.purpose,
             task_type=spec.task_type, compute_target=spec.compute_target)
    # Cloud jobs are not drained by the local worker; hand them to the cloud dispatch seam.
    if spec.compute_target == "cloud":
        from services.training.cloud import mark_queued_for_cloud

        await mark_queued_for_cloud(job_id)
    return str(job_id)


async def _bump(job_id, **fields) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(str(job_id)))
        if not j:
            return
        for k, v in fields.items():
            setattr(j, k, v)
        await db.commit()


async def _apply_progress(job_id, d: dict) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(str(job_id)))
        if not j:
            return
        if "stage" in d:
            j.stage = d["stage"]
        if "epoch" in d:
            counts = dict(j.counts or {})
            counts["epoch"] = d["epoch"]
            if d.get("total_epochs"):
                counts["total_epochs"] = d["total_epochs"]
            j.counts = counts
            tot = d.get("total_epochs") or 1
            j.progress = round(0.15 + 0.70 * (d["epoch"] / tot), 3)  # train spans 15%..85%
        if "metrics" in d:
            m = dict(j.metrics or {})
            m["live"] = d["metrics"]
            # Keep the whole curve, not only the latest point. Only the last epoch was retained, so the
            # shape of a run (early plateau, late collapse, the epoch a metric peaked) was unreadable
            # afterwards without parsing ultralytics' CSV off disk, which does not survive a cleanup.
            curve = list(m.get("curve") or [])
            if "epoch" in d:
                curve.append({"epoch": d["epoch"], **d["metrics"]})
                # Bound it: a long sweep should not grow a row without limit. The last 500 epochs is far
                # more than any run here, and dropping from the front keeps the recent shape.
                m["curve"] = curve[-500:]
            j.metrics = m
        await db.commit()


async def _is_canceled(job_id) -> bool:
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(str(job_id)))
        return bool(j and j.cancel_requested)


async def run_job(job_id) -> dict:
    settings = get_settings()
    onto = get_ontology()
    loop = asyncio.get_running_loop()
    pq: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def progress(d: dict) -> None:
        # thread-safe: callable from the executor (train) thread or the loop (build)
        loop.call_soon_threadsafe(pq.put_nowait, d)

    async def drain() -> None:
        while True:
            d = await pq.get()
            if d is None:
                return
            try:
                await _apply_progress(job_id, d)
            except Exception as exc:  # noqa: BLE001
                log.warning("training.progress_failed", error=str(exc))
            if await _is_canceled(job_id):
                cancel.set()

    drain_task = spawn(drain(), name="drain")
    try:
        async with get_sessionmaker()() as db:
            job = await db.get(TrainingJob, uuid.UUID(str(job_id)))
            if job is None:
                raise RuntimeError(f"training job {job_id} not found")
            spec = TrainJobSpec(**job.config)

        await _bump(job_id, status="running", stage="build", progress=0.05)
        if await _is_canceled(job_id):
            await _bump(job_id, status="canceled", stage="canceled")
            return {"canceled": True}

        task = get_task(spec.task_type)
        name = _safe_name(spec.purpose, job_id)
        base_weights = spec.base_weights or task.default_base_weights()
        imgsz = int(spec.hparams.get("imgsz", settings.training.default_imgsz))

        ds = await task.build_dataset({"name": name, "dataset_spec": spec.dataset_spec}, progress)
        if ds["n_train_images"] < 4 or ds["n_val_images"] < 1:
            raise RuntimeError(f"not enough data: {ds['n_train_images']} train / {ds['n_val_images']} val")

        # What this model is actually trained on, as a content hash rather than a name. `dataset_commit`
        # used to carry the build name, so two "loop-v1" builds a week apart were different data with the
        # same label and a promoted champion could not be traced to the corpus behind it.
        fingerprint = dataset_fingerprint(ds["dir"], str(ds.get("ontology_version") or ""),
                                          spec.dataset_spec)
        dataset_commit = f"{name}+{fingerprint}" if fingerprint else name
        await _bump(job_id, stage="evaluate", progress=0.10,
                    counts={"n_train": ds["n_train_images"], "n_val": ds["n_val_images"], "classes": ds["classes"]})

        baseline = await loop.run_in_executor(None, task.evaluate, base_weights, ds["data_yaml"], imgsz)
        await _bump(job_id, stage="train", progress=0.15, metrics={"baseline": _slim(baseline)})

        hparams = {**spec.hparams, "name": name, "_should_stop": cancel.is_set}
        weights = await loop.run_in_executor(None, task.train, ds["data_yaml"], base_weights, hparams, progress)

        if cancel.is_set():
            await _bump(job_id, status="canceled", stage="canceled", progress=0.9)
            return {"canceled": True}

        await _bump(job_id, stage="evaluate", progress=0.88)
        candidate = await loop.run_in_executor(None, task.evaluate, weights, ds["data_yaml"], imgsz)
        # Safe-mIoU so the challenger carries a safety score the champion gate requires (fail-closed).
        try:
            from services.training.eval import safe_miou_report

            sm = await loop.run_in_executor(None, safe_miou_report, weights, ds["data_yaml"], "val", imgsz)
            if sm.get("safe_miou") is not None:
                candidate["safe_miou"] = sm["safe_miou"]
        except Exception as exc:  # noqa: BLE001
            log.warning("training.safe_miou_failed", job_id=str(job_id), error=str(exc))
        gate = task.gate(candidate, baseline, spec.gate)

        store = get_object_store()
        store.ensure_bucket()
        weights_uri = await loop.run_in_executor(
            None, store.put_file, f"models/{name}/best.pt", weights, "application/octet-stream")

        epochs = int(spec.hparams.get("epochs", settings.training.default_epochs))
        run_id = _run_id(name, ds, epochs)
        # The training gate compares this candidate against its own baseline. Promotion is a separate,
        # stricter question answered against the gold set by the champion gate, so this flag records only
        # that promotion was asked for and cleared the first hurdle. Whether it happened is decided below,
        # after the run is registered, and written back over this value.
        promotion_requested = spec.promote and gate["promote"]
        promotion: dict | None = None
        async with get_sessionmaker()() as db:
            await db.merge(ModelRun(
                run_id=run_id, base_weights=base_weights, weights_uri=weights_uri, dataset_name=name,
                n_train=ds["n_train_images"], n_val=ds["n_val_images"], epochs=epochs,
                metrics=candidate, baseline_metrics=baseline, gate=gate, promoted=False,
                ontology_version=onto.version, purpose=spec.purpose, task_type=spec.task_type,
                job_id=uuid.UUID(str(job_id)), notes=spec.notes or f"task={spec.task_type}",
            ))
            await db.commit()
            # Auto-register as a non-champion challenger so the controller can see and gate it; this is
            # the seam that lets the loop close (train -> register -> champion gate -> serve) on its own.
            try:
                from services.govern.registry import register

                await record_dataset_commit(db, dataset_commit, build=ds, spec=spec.dataset_spec,
                                            fingerprint=fingerprint)
                await register(db, run_id, spec.task_type, candidate or {},
                               dataset_commit=dataset_commit,
                               weights_uri=weights_uri, notes=f"auto-registered from job {job_id}")
            except Exception as exc:  # noqa: BLE001
                log.warning("registry.auto_register_failed", job_id=str(job_id), run_id=run_id, error=str(exc))

            if promotion_requested:
                promotion = await _promote_run(db, run_id, spec.task_type)

        do_promote = bool(promotion and promotion.get("promoted"))
        if do_promote:
            async with get_sessionmaker()() as db:
                run = await db.get(ModelRun, run_id)
                if run is not None:
                    run.promoted = True
                    await db.commit()

        result = {
            "run_id": run_id, "weights_uri": weights_uri, "gate": gate, "promoted": do_promote,
            "baseline_map50": baseline.get("map50"), "candidate_map50": candidate.get("map50"),
        }
        if promotion is not None:
            # The whole champion verdict, so a blocked promotion says which metric refused it rather than
            # leaving a run that asked to be promoted and silently was not.
            result["promotion"] = promotion
        await _bump(job_id, status="done", stage="done", progress=1.0, run_id=run_id, result=result,
                    metrics={"baseline": _slim(baseline), "candidate": _slim(candidate)})
        log.info("training.done", job_id=str(job_id), run_id=run_id, promoted=do_promote,
                 candidate_map50=candidate.get("map50"))
        return result
    finally:
        await pq.put(None)
        await drain_task


async def run_job_guarded(job_id) -> None:
    try:
        await run_job(job_id)
    except Exception as exc:  # noqa: BLE001
        log.error("training.failed", job_id=str(job_id), error=str(exc))
        await _bump(job_id, status="error", error=str(exc))
