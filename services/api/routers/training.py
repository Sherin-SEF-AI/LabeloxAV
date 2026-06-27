"""Training endpoints: enqueue jobs, poll status, list jobs/tasks, cancel, and view the model
registry. The API only enqueues (writes a pending training_job row); the out-of-process worker
(make train-worker) executes on the GPU, so these handlers never block the API event loop.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from db.models import ModelRun, TrainingJob
from db.session import get_sessionmaker
from services.api.deps import TrainingStartIn
from services.training.jobs import TrainJobSpec, enqueue_job
from services.training.tasks import get_task, list_tasks

router = APIRouter()


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
async def start(payload: TrainingStartIn):
    try:
        get_task(payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    job_id = await enqueue_job(TrainJobSpec(**payload.model_dump()))
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


@router.get("/training/{job_id}")
async def status(job_id: str):
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
    if j is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return _job_dict(j)


@router.get("/training")
async def list_jobs(limit: int = 50):
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit))).scalars().all()
    return [_job_dict(j) for j in rows]


@router.post("/training/{job_id}/cancel")
async def cancel(job_id: str):
    async with get_sessionmaker()() as db:
        j = await db.get(TrainingJob, uuid.UUID(job_id))
        if j is None:
            raise HTTPException(status_code=404, detail="training job not found")
        if j.status == "pending":
            j.status = "canceled"
        else:
            j.cancel_requested = True  # honored by the worker at the next epoch boundary
        await db.commit()
        return _job_dict(j)
