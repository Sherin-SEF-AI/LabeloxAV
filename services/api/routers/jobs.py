"""Unified jobs view: import, training, and autolabel jobs in one normalized stream for the Jobs
dashboard (the single place to watch everything the engine is doing).

Cancel lives here rather than beside each job's own start route, for the same reason the list does: this is
where a person looks when something is stuck, and until now the only cancel among 610 endpoints was
training's. Everything else needed an UPDATE against the database, which was worse than an inconvenience:
the autolabel router refuses to start while any row reads `running`, so one unwanted job blocked the whole
deployment.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AutolabelJob, ExportJob, ImportJob, MapFusionJob, RelabelJob, TrainingJob
from services.api.deps import db_session
from services.job_control import QUEUED_CLOUD

router = APIRouter()

# The kind names this router already emits, mapped back to their tables. Training is deliberately absent:
# it is drained by a separate worker holding a GPU lease, so it cancels through its own route
# (`services/api/routers/training.py`) which knows to ask the worker rather than to write the row.
CANCELABLE = {
    "autolabel": AutolabelJob,
    "import": ImportJob,
    "export": ExportJob,
    "relabel": RelabelJob,
    "map_fusion": MapFusionJob,
}


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _import_label(source_uri: str | None, fmt: str | None) -> str:
    """What to call an import row: the file it came from, not the format it is in.

    Every import was labelled by its format, so a folder of 186 clips produced 186 rows all reading "video".
    The page exists to tell one run from another and that label made it impossible. The filename is already
    in source_uri; the format is still the fallback for an import that has no file behind it.
    """
    if source_uri:
        # Drop the scheme before looking for a basename, or a bare "s3://" yields "s3:" and a job row is
        # labelled with a fragment of its own URL.
        path = source_uri.split("://", 1)[-1] if "://" in source_uri else source_uri
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name:
            return name
    return fmt or "import"


@router.get("/jobs")
async def jobs(limit: int = 100, db: AsyncSession = Depends(db_session)):
    limit = min(max(limit, 1), 1000)
    out: list[dict] = []

    for j in (await db.execute(select(ImportJob).order_by(ImportJob.created_at.desc()).limit(limit))).scalars():
        c = j.counts or {}
        out.append({"job_id": str(j.job_id), "kind": "import", "status": j.status, "progress": j.progress,
                    "label": _import_label(j.source_uri, j.format),
                    "detail": f"{c.get('frames', 0)}fr / {c.get('objects', 0)}obj",
                    "link": "/import", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit))).scalars():
        c = j.counts or {}
        ep = f"ep {c.get('epoch', 0)}/{c.get('total_epochs', 0)}" if c.get("total_epochs") else (j.stage or "")
        out.append({"job_id": str(j.job_id), "kind": "training", "status": j.status, "progress": j.progress,
                    "label": f"{j.purpose} ({j.compute_target})", "detail": ep, "link": "/training", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(AutolabelJob).order_by(AutolabelJob.created_at.desc()).limit(limit))).scalars():
        c = j.counts or {}
        cloud = c.get("compute_target") == "cloud"
        # The status is read straight from the row now. It used to be translated here, because the row said
        # `pending` while the start response said `queued-cloud`, so this list was the only place the two
        # agreed and everything reading the database directly disagreed with both.
        out.append({"job_id": str(j.job_id), "kind": "autolabel",
                    "status": j.status, "progress": j.progress,
                    "label": str(j.session_id)[:8],
                    "detail": "queued for cloud A100 (SAM3.1+Qwen3-VL+YOLO26)" if cloud
                              else f"{c.get('n_frames', 0)}fr / {c.get('n_objects', 0)}obj",
                    "link": "/", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(ExportJob).order_by(ExportJob.created_at.desc()).limit(limit))).scalars():
        out.append({"job_id": str(j.job_id), "kind": "export", "status": j.status, "progress": j.progress,
                    "label": j.name, "detail": f"{j.object_count} obj" + (f" -> {j.commit_id[:12]}" if j.commit_id else ""),
                    "link": "/datasets", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(MapFusionJob).order_by(MapFusionJob.created_at.desc()).limit(limit))).scalars():
        cloud = j.compute_target == "cloud"
        c = j.counts or {}
        out.append({"job_id": str(j.job_id), "kind": "map_fusion",
                    "status": j.status, "progress": j.progress,
                    "label": f"{j.region} ({len(j.session_ids or [])} drives)",
                    "detail": "queued for cloud A100 (GTSAM)" if cloud and j.status == QUEUED_CLOUD
                              else (j.stage or "") + (f" -> {j.commit_id}" if j.commit_id else "")
                              + (f" {c.get('fused', '')}el" if c.get("fused") else ""),
                    "link": "/map", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(RelabelJob).order_by(RelabelJob.created_at.desc()).limit(limit))).scalars():
        cloud = j.compute_target == "cloud"
        r = j.result or {}
        out.append({"job_id": str(j.job_id), "kind": "relabel",
                    "status": j.status, "progress": j.progress,
                    "label": j.model_version[:16],
                    "detail": "queued for cloud A100 (champion re-inference)" if cloud and j.status == QUEUED_CLOUD
                              else (j.stage or "") + (f" {r.get('applied', '')} applied" if r.get("applied") is not None else ""),
                    "link": "/govern", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out[:limit]


@router.post("/jobs/{kind}/{job_id}/cancel")
async def cancel(kind: str, job_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Cancel a job of any kind. Answers whether it stopped or was only asked to.

    A job that never started, or whose process is already gone, is stopped the moment the row changes. A
    live one stops at its next checkpoint, which is a frame for auto-label and a hundred frames for import.
    The two are reported separately because an operator waiting for the GPU cares which one they got.
    """
    from services.job_control import cancel_job

    model = CANCELABLE.get(kind)
    if model is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown job kind {kind!r}; expected one of {sorted(CANCELABLE)}"
                                   + (" (training cancels through /api/training/{id}/cancel)"
                                      if kind == "training" else ""))
    return await cancel_job(db, model, job_id)
