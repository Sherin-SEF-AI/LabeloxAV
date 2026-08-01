"""Import endpoints: kick off a background import of an uploaded/located dataset and poll its status.

The job is recorded in Postgres (import_job) and run as an asyncio task in the API process - the
simplest robust model on one box; the Redpanda-consumer worker (TOPIC_IMPORT_REQUESTED) is the
documented cloud seam for horizontal scale and is intentionally not built here.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from db.models import ImportJob
from db.session import get_sessionmaker
from services.api.deps import ImportStartIn
from services.imports.records import ImportSpec
from services.imports.run import ALL_FORMATS, run_import_guarded

router = APIRouter()


@router.get("/ingest/progress")
async def ingest_progress():
    """Live ingest progress.

    Read from the ImportJob rows first and the batch log only as a fallback. It used to be the log alone,
    scraped with a regex, which silently reports a quiet system the moment the API runs in a different
    container from the script rather than failing in a way anyone would notice.
    """
    from services.ingest.progress import read_progress, with_frame_count

    async with get_sessionmaker()() as db:
        progress = await read_progress(db)
        return await with_frame_count(db, progress)


def _job_dict(j: ImportJob) -> dict:
    return {
        "job_id": str(j.job_id), "status": j.status, "format": j.format,
        "source_uri": j.source_uri, "target_vehicle": j.target_vehicle, "city": j.city,
        "progress": j.progress, "counts": j.counts, "error": j.error,
        "session_id": str(j.session_id) if j.session_id else None,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "updated_at": j.updated_at.isoformat() if j.updated_at else None,
    }


@router.post("/imports/dry-run")
async def dry_run(payload: ImportStartIn):
    """Parse a source and report what importing it would cost, without writing anything.

    A migration is the moment a customer finds out what their taxonomy survives. Telling them afterwards
    that two of their classes became one of ours invites them to discover the loss a month later; telling
    them here makes it a decision they get to make. Read-only by construction: it parses and remaps in
    memory and touches no table.
    """
    from services.imports.conflicts import taxonomy_report
    from services.imports.run import ADAPTERS, _acquire_source

    if payload.format not in ADAPTERS:
        raise HTTPException(status_code=400,
                            detail=f"'{payload.format}' has no annotation adapter to dry-run; "
                                   f"choose one of {sorted(ADAPTERS)}")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            root = _acquire_source(payload.source_uri, Path(tmp))
            frames = ADAPTERS[payload.format](root)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"could not read that source: {exc}") from exc

    return {"format": payload.format, "source_uri": payload.source_uri,
            "frames": len(frames),
            "report": taxonomy_report(frames)}


@router.post("/imports/start")
async def start(payload: ImportStartIn):
    if payload.format not in ALL_FORMATS:
        raise HTTPException(status_code=400, detail=f"unknown format {payload.format}; choose {ALL_FORMATS}")
    job_id = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(ImportJob(job_id=job_id, status="pending", format=payload.format, source_uri=payload.source_uri,
                         target_vehicle=payload.target_vehicle, city=payload.city, progress=0.0, counts={}))
        await db.commit()
    spec = ImportSpec(format=payload.format, source_uri=payload.source_uri,
                      target_vehicle=payload.target_vehicle, city=payload.city, options=payload.options)
    asyncio.create_task(run_import_guarded(spec, job_id))
    return {"job_id": str(job_id), "status": "pending"}


@router.get("/imports/{job_id}")
async def status(job_id: str):
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, uuid.UUID(job_id))
    if j is None:
        raise HTTPException(status_code=404, detail="import job not found")
    return _job_dict(j)


@router.get("/imports")
async def list_jobs(limit: int = 50):
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(ImportJob).order_by(ImportJob.created_at.desc()).limit(limit))).scalars().all()
    return [_job_dict(j) for j in rows]
