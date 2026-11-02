"""Error-detection endpoints (M4.1): run the detectors, list the ranked error-candidate queue, and
confirm or dismiss candidates (confirming feeds the correction and retrain path)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import current_user, db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.errordetect.queue import (
    bulk_verdict,
    confirm_error,
    detector_precision,
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
async def candidates(status: str = "pending", limit: int = 100, kind: str | None = None,
                     db: AsyncSession = Depends(db_session), hidden=Depends(caller_hidden_projects)):
    """The ranked queue, optionally one detector at a time.

    Filtering by kind matters more than it looks. The queue ranks across detectors by score, and the
    detectors do not produce commensurable scores, so a mixed page is mostly whichever detector emits the
    largest numbers. Reviewing one detector at a time is also the only way to judge that detector, which is
    what the verdicts are for.
    """
    from services.project_access import drop_hidden

    return await drop_hidden(db, hidden, await list_candidates(db, status, limit, kind=kind), "object_id")


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
async def dismiss(candidate_id: str, note: str | None = None,
                  db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    return await dismiss_error(db, candidate_id, user_id=str(user.user_id) if user else None, note=note)


class BulkVerdictIn(BaseModel):
    candidate_ids: list[str]
    verdict: str                      # confirmed_error | dismissed
    note: str | None = None
    apply_proposed: bool = True
    reviewer: str = "error-detect"


@router.post("/errordetect/candidates/bulk")
async def bulk(payload: BulkVerdictIn, db: AsyncSession = Depends(db_session),
               user=Depends(current_user)):
    """Rule on many candidates in one action.

    298,529 candidates carry one verdict between them, and confirm and dismiss both took a single id, so the
    queue was not reviewable in principle. Dismissing 400 near-duplicate candidates is one judgement about a
    detector rather than 400 judgements about objects, and `note` is where that judgement survives.

    Candidates somebody already ruled on are skipped rather than overwritten, and reported as
    already_decided: two reviewers on the same queue is normal, and silently replacing the first verdict
    would corrupt the calibration with no trace of the disagreement.
    """
    return await bulk_verdict(db, payload.candidate_ids, payload.verdict,
                              user_id=str(user.user_id) if user else None,
                              reviewer=payload.reviewer, note=payload.note,
                              apply_proposed=payload.apply_proposed)


@router.get("/errordetect/precision")
async def precision(db: AsyncSession = Depends(db_session)):
    """Per-detector precision, from the verdicts humans have given.

    The calibration this queue has never had. Each detector emits a score and nothing said whether a high
    one meant anything. Confirmed over confirmed-plus-dismissed is the answer, and every rate carries its
    interval and count so a detector with three verdicts is not compared against one with three hundred.
    """
    return await detector_precision(db)


class BenchmarkIn(BaseModel):
    session_ids: list[str] | None = None
    rate: float = 0.1


@router.post("/errordetect/benchmark")
async def benchmark(payload: BenchmarkIn, db: AsyncSession = Depends(db_session),
                    _admin=Depends(require_role("admin"))):
    """How well the class-error detectors find errors, measured by planting them (services/errordetect/
    benchmark.py). Runs in one transaction that is rolled back: no label changes. Without session ids, the
    three sessions with the most accepted, embedded labels. Administrators only, since it reads across
    projects."""
    from fastapi import HTTPException

    from services.errordetect.benchmark import benchmark_sessions, default_sessions

    if not 0.01 <= payload.rate <= 0.5:
        raise HTTPException(400, "rate is the share of labels to plant errors in, between 0.01 and 0.5")
    sessions = payload.session_ids or await default_sessions(db, 3)
    if not sessions:
        raise HTTPException(404, "no session has accepted labels with embeddings yet")
    return await benchmark_sessions(db, sessions[:10], rate=payload.rate)
