"""Unified jobs view: import, training, and autolabel jobs in one normalized stream for the Jobs
dashboard (the single place to watch everything the engine is doing)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AutolabelJob, ExportJob, ImportJob, MapFusionJob, RelabelJob, TrainingJob
from services.api.deps import db_session

router = APIRouter()


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


@router.get("/jobs")
async def jobs(limit: int = 100, db: AsyncSession = Depends(db_session)):
    limit = min(max(limit, 1), 1000)
    out: list[dict] = []

    for j in (await db.execute(select(ImportJob).order_by(ImportJob.created_at.desc()).limit(limit))).scalars():
        c = j.counts or {}
        out.append({"job_id": str(j.job_id), "kind": "import", "status": j.status, "progress": j.progress,
                    "label": j.format, "detail": f"{c.get('frames', 0)}fr / {c.get('objects', 0)}obj",
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
        out.append({"job_id": str(j.job_id), "kind": "autolabel",
                    "status": "queued-cloud" if cloud else j.status, "progress": j.progress,
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
                    "status": "queued-cloud" if cloud and j.status == "pending" else j.status, "progress": j.progress,
                    "label": f"{j.region} ({len(j.session_ids or [])} drives)",
                    "detail": "queued for cloud A100 (GTSAM)" if cloud and j.status == "pending"
                              else (j.stage or "") + (f" -> {j.commit_id}" if j.commit_id else "")
                              + (f" {c.get('fused', '')}el" if c.get("fused") else ""),
                    "link": "/map", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    for j in (await db.execute(select(RelabelJob).order_by(RelabelJob.created_at.desc()).limit(limit))).scalars():
        cloud = j.compute_target == "cloud"
        r = j.result or {}
        out.append({"job_id": str(j.job_id), "kind": "relabel",
                    "status": "queued-cloud" if cloud and j.status == "pending" else j.status, "progress": j.progress,
                    "label": j.model_version[:16],
                    "detail": "queued for cloud A100 (champion re-inference)" if cloud and j.status == "pending"
                              else (j.stage or "") + (f" {r.get('applied', '')} applied" if r.get("applied") is not None else ""),
                    "link": "/govern", "error": j.error,
                    "created_at": _iso(j.created_at), "updated_at": _iso(j.updated_at)})

    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out[:limit]
