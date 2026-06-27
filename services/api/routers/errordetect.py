"""Error-detection endpoints (M4.1): run the detectors, list the ranked error-candidate queue, and
confirm or dismiss candidates (confirming feeds the correction and retrain path)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.errordetect.queue import (
    confirm_error,
    dismiss_error,
    list_candidates,
    run_detection,
    summary,
)

router = APIRouter()


class DetectIn(BaseModel):
    session_id: str | None = None
    kinds: list[str] | None = None


@router.post("/errordetect/run")
async def run(payload: DetectIn, db: AsyncSession = Depends(db_session)):
    return await run_detection(db, payload.session_id, payload.kinds)


@router.get("/errordetect/candidates")
async def candidates(status: str = "pending", limit: int = 100, db: AsyncSession = Depends(db_session)):
    return await list_candidates(db, status, limit)


@router.get("/errordetect/summary")
async def summary_ep(db: AsyncSession = Depends(db_session)):
    return await summary(db)


class ConfirmIn(BaseModel):
    apply_proposed: bool = True
    reviewer: str = "error-detect"
    user_id: str | None = None


@router.post("/errordetect/candidates/{candidate_id}/confirm")
async def confirm(candidate_id: str, payload: ConfirmIn, db: AsyncSession = Depends(db_session)):
    return await confirm_error(db, candidate_id, payload.apply_proposed, payload.reviewer, payload.user_id)


@router.post("/errordetect/candidates/{candidate_id}/dismiss")
async def dismiss(candidate_id: str, db: AsyncSession = Depends(db_session)):
    return await dismiss_error(db, candidate_id)
