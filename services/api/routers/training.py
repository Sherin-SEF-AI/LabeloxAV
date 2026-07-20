"""Training endpoints: enqueue jobs, poll status, list jobs/tasks, cancel, and view the model
registry. The API only enqueues (writes a pending training_job row); the out-of-process worker
(make train-worker) executes on the GPU, so these handlers never block the API event loop.
"""

from __future__ import annotations

import csv
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from core.config import get_settings
from db.models import ModelRun, TrainingJob
from db.session import get_sessionmaker
from services.api.deps import TrainingStartIn, current_user, role_rank
from services.training.jobs import TrainJobSpec, enqueue_job
from services.training.tasks import get_task, list_tasks

router = APIRouter()

# results.csv column -> the short key we expose for per-epoch curves
_CURVE_COLS = {
    "train/box_loss": "train_box_loss", "train/cls_loss": "train_cls_loss",
    "val/box_loss": "val_box_loss", "val/cls_loss": "val_cls_loss",
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map",
}


def _run_dict(r: ModelRun) -> dict:
    """A model run with enough for a graphical baseline->candidate comparison + the gate decision."""
    return {
        "run_id": r.run_id, "dataset_name": r.dataset_name, "purpose": r.purpose,
        "epochs": r.epochs, "n_train": r.n_train, "n_val": r.n_val, "promoted": r.promoted,
        "base_weights": r.base_weights, "notes": r.notes,
        "baseline_metrics": r.baseline_metrics or {}, "metrics": r.metrics or {}, "gate": r.gate or {},
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _job_dict(j: TrainingJob) -> dict:
    return {
        "job_id": str(j.job_id), "status": j.status, "purpose": j.purpose, "task_type": j.task_type,
        "compute_target": j.compute_target,
        "stage": j.stage, "progress": j.progress, "counts": j.counts, "metrics": j.metrics,
        "result": j.result, "error": j.error, "run_id": j.run_id, "config": j.config,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "updated_at": j.updated_at.isoformat() if j.updated_at else None,
    }


@router.get("/training/tasks")
async def tasks():
    return list_tasks()


@router.post("/training/start")
async def start(payload: TrainingStartIn, user=Depends(current_user)):
    try:
        get_task(payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    spec = payload.model_dump()
    # C7: auto-promotion is a governance action (equivalent to /govern/promote, which is admin-only). A
    # reviewer may enqueue training, but only an admin may request that its result promote a champion.
    if spec.get("promote") and role_rank(getattr(user, "role", None)) < role_rank("admin"):
        raise HTTPException(status_code=403, detail="promote requires admin role")
    job_id = await enqueue_job(TrainJobSpec(**spec))
    return {"job_id": job_id, "status": "pending"}


@router.get("/training/registry")
async def registry():
    """Model lines grouped by purpose; the active model per line is its latest promoted run."""
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(ModelRun).order_by(ModelRun.created_at.desc()))).scalars().all()
    lines: dict[str, dict] = {}
    for r in rows:
        line = lines.setdefault(r.purpose, {"purpose": r.purpose, "task_type": r.task_type, "runs": [], "promoted": None})
        entry = {
            "run_id": r.run_id, "dataset_name": r.dataset_name, "epochs": r.epochs,
            "map50": (r.metrics or {}).get("map50"), "safe_miou": (r.metrics or {}).get("safe_miou"),
            "promoted": r.promoted, "weights_uri": r.weights_uri,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        line["runs"].append(entry)
        if r.promoted and line["promoted"] is None:
            line["promoted"] = entry
    return list(lines.values())


# NB: these literal-prefixed routes must precede /training/{job_id} (a uuid path param) so "runs"
# is not matched (and 422'd) as a job id.
@router.get("/training/runs")
async def runs(limit: int = 50):
    """Recorded model runs (finetune close-the-loop output), newest first, with baseline vs candidate
    metrics and the promotion gate - the data behind the graphical comparison."""
    limit = min(max(limit, 1), 500)
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit))).scalars().all()
    return [_run_dict(r) for r in rows]


@router.get("/training/runs/{run_id}/curve")
async def run_curve(run_id: str):
    """Per-epoch training curves parsed from ultralytics results.csv for this run's dataset."""
    async with get_sessionmaker()() as db:
        r = await db.get(ModelRun, run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="model run not found")
    path = get_settings().scratch_path() / "training" / "runs" / r.dataset_name / "results.csv"
    if not path.exists():
        return {"run_id": run_id, "epochs": [], "series": {}, "available": False}
    epochs: list[int] = []
    series: dict[str, list[float]] = {v: [] for v in _CURVE_COLS.values()}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): v for k, v in row.items()}
            try:
                epochs.append(int(float(row["epoch"])))
            except (KeyError, ValueError):
                continue
            for col, key in _CURVE_COLS.items():
                try:
                    series[key].append(round(float(row[col]), 5))
                except (KeyError, ValueError):
                    series[key].append(0.0)
    return {"run_id": run_id, "dataset_name": r.dataset_name, "epochs": epochs, "series": series,
            "available": bool(epochs)}


@router.get("/training/{job_id}")
async def status(job_id: uuid.UUID):
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return _job_dict(j)


@router.get("/training")
async def list_jobs(limit: int = 50):
    limit = min(max(limit, 1), 1000)
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit))).scalars().all()
    return [_job_dict(j) for j in rows]


@router.post("/training/{job_id}/cancel")
async def cancel(job_id: uuid.UUID):
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="training job not found")
        if j.status == "pending":
            j.status = "canceled"
        else:
            j.cancel_requested = True  # honored by the worker at the next epoch boundary
        await db.commit()
        return _job_dict(j)
