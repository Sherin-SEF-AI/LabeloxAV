"""Active-learning curation endpoints: the smart "what to label next" surface. Read summaries (novel
frames = coverage gaps, near-duplicates = skip, diversity sample) over DINOv2 frame embeddings, and
trigger embedding as a background task (GPU-light; yields to training)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import TrainingJob
from services.analytics.curation import curation_summary, diverse_sample
from services.api.deps import db_session

log = get_logger("api_curation")
router = APIRouter()


@router.get("/curation/summary")
async def summary(session_id: str | None = None):
    return await curation_summary(session_id)


@router.post("/curation/extract")
async def extract(session_id: str):
    """Intelligent frame extraction: re-select a session keeping scene changes + rare events, dropping
    near-static stretches, within a frame budget. Writes selected + novelty_score, never deletes."""
    from uuid import UUID as _UUID

    from services.ingest.extract_smart import smart_select_session

    return await smart_select_session(_UUID(session_id))


@router.post("/curation/dedup")
async def dedup(session_id: str):
    """Near-duplicate detection for a session (pHash prefilter + DINOv3 confirm): groups near-dups, keeps
    one canonical per group, sets selected=false on the rest. CPU + existing vectors, no GPU needed."""
    from uuid import UUID as _UUID

    from services.intelligence.dedup import dedup_session

    return await dedup_session(_UUID(session_id))


@router.get("/curation/diverse")
async def diverse(session_id: str | None = None, k: int = 50):
    return await diverse_sample(session_id, k)


async def _embed_guarded(session_id) -> None:
    from uuid import UUID

    from services.intelligence.embed.service import embed_frames

    try:
        await embed_frames(UUID(session_id) if session_id else None)
    except Exception as exc:  # noqa: BLE001
        log.error("curation.embed_failed", error=str(exc))


@router.post("/curation/embed")
async def embed(session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
        raise HTTPException(503, "GPU reserved for a training job; embedding is paused until it finishes")
    asyncio.create_task(_embed_guarded(session_id))
    return {"started": True}
