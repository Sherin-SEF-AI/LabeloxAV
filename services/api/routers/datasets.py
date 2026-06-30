"""Datasets + delivery: seal a versioned dataset (DatasetCommit), run the export as a background job,
and hand back presigned download links. This is how labeled data leaves the engine as a product."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import DatasetCommit, ExportJob
from db.session import get_sessionmaker
from services.api.deps import ExportIn, db_session

log = get_logger("api_datasets")
router = APIRouter()


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


async def _bump(job_id, **fields) -> None:
    async with get_sessionmaker()() as db:
        j = await db.get(ExportJob, uuid.UUID(str(job_id)))
        if j:
            for k, v in fields.items():
                setattr(j, k, v)
            await db.commit()


async def _run_export(job_id, spec) -> None:
    from services.export.dataset import export_dataset

    await _bump(job_id, status="running", progress=0.1)
    try:
        result = await export_dataset(spec)
        await _bump(job_id, status="done", progress=1.0, commit_id=result["commit_id"], object_count=result["object_count"])
        log.info("export.job_done", job_id=str(job_id), commit_id=result["commit_id"])
    except Exception as exc:  # noqa: BLE001
        log.error("export.job_failed", job_id=str(job_id), error=str(exc))
        await _bump(job_id, status="error", error=str(exc))


@router.post("/datasets/export")
async def start_export(payload: ExportIn, db: AsyncSession = Depends(db_session)):
    from services.export.dataset import SliceSpec

    spec = SliceSpec(name=payload.name, states=payload.states, class_names=payload.class_names,
                     cities=payload.cities, session_id=payload.session_id, min_conf=payload.min_conf,
                     formats=payload.formats, limit=payload.limit)
    job_id = uuid.uuid4()
    db.add(ExportJob(job_id=job_id, name=payload.name, spec=spec.model_dump(), status="pending"))
    await db.commit()
    asyncio.create_task(_run_export(job_id, spec))
    return {"job_id": str(job_id), "status": "pending"}


@router.get("/datasets")
async def list_datasets(limit: int = 100, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(DatasetCommit).order_by(DatasetCommit.created_at.desc()).limit(limit))).scalars().all()
    return [
        {"commit_id": d.commit_id, "name": (d.slice_spec or {}).get("name"),
         "object_count": d.object_count, "formats": (d.slice_spec or {}).get("formats", []),
         "ontology_version": d.ontology_version, "n_files": len(d.export_uris or {}),
         "created_at": _iso(d.created_at)}
        for d in rows
    ]


@router.get("/datasets/{commit_id}/lineage")
async def dataset_lineage(commit_id: str):
    """Milestone I: walk a snapshot's parent chain back to its root (the slice's version history)."""
    from services.export.snapshots import lineage
    res = await lineage(commit_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/datasets/{a_id}/diff/{b_id}")
async def dataset_diff(a_id: str, b_id: str):
    """Milestone I: what changed between two snapshots (count deltas, ontology change, slice-spec changes)."""
    from services.export.snapshots import compare_commits
    res = await compare_commits(a_id, b_id)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    return res


@router.get("/datasets/{commit_id}")
async def dataset_detail(commit_id: str, db: AsyncSession = Depends(db_session)):
    d = await db.get(DatasetCommit, commit_id)
    if d is None:
        raise HTTPException(404, "dataset not found")
    store = get_object_store()
    files = []
    for path, uri in (d.export_uris or {}).items():
        try:
            files.append({"path": path, "url": store.presigned_get(uri)})
        except Exception:  # noqa: BLE001
            files.append({"path": path, "url": None})
    return {
        "commit_id": d.commit_id, "name": (d.slice_spec or {}).get("name"),
        "object_count": d.object_count, "formats": (d.slice_spec or {}).get("formats", []),
        "ontology_version": d.ontology_version, "created_at": _iso(d.created_at),
        "slice_spec": d.slice_spec, "files": sorted(files, key=lambda f: f["path"]),
    }
