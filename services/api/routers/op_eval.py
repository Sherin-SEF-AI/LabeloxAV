"""Measured precision per batch operation, or an explicit refusal to guess.

The contract the client depends on: a 200 carries a real measurement, a 404 means unmeasured. There is no
third state and no default number, because a batch button that shows a plausible-looking precision nobody
computed is worse than one that admits it does not know. On a 404 the client makes the dry run mandatory and
routes everything to review, which is the product behaviour, not a degraded mode.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.training.op_precision import (
    OPERATION_KINDS,
    WINDOW_DAYS,
    measure_all,
    measure_operation,
)

router = APIRouter()


def _window_start(window_days: int | None) -> int | None:
    """The oldest run a score should consider, or None for all time (window_days=0)."""
    from core.timebase import now_ns

    if not window_days:
        return None
    return now_ns() - int(window_days) * 86_400 * 1_000_000_000


@router.get("/eval/operations", dependencies=[Depends(require_role("annotator"))])
async def operations(db: AsyncSession = Depends(db_session),
                     window_days: int | None = WINDOW_DAYS) -> dict:
    """Every operation kind with its measurement state, for the panel header.

    Scored over recent runs by default, so an operation that has been fixed can earn its score back. Pass
    window_days=0 for the all-time view, which is the right question when auditing what an operation has
    ever done rather than whether to trust it now.
    """
    window = None if not window_days else int(window_days)
    return {"operations": await measure_all(db, window_days=window), "kinds": list(OPERATION_KINDS),
            "window_days": window}


@router.get("/eval/operations/{op_type}/latest", dependencies=[Depends(require_role("annotator"))])
async def operation_latest(op_type: str, db: AsyncSession = Depends(db_session),
                           window_days: int | None = WINDOW_DAYS) -> dict:
    """The latest measurement for one operation, or 404 when there is not enough evidence to have one."""
    if op_type not in OPERATION_KINDS:
        raise HTTPException(status_code=404, detail=f"unknown operation type {op_type!r}")
    r = await measure_operation(db, op_type, runs_since_ns=_window_start(window_days))
    if not r.get("measured"):
        # 404 rather than a 200 carrying nulls: the client's unmeasured path is the same one it uses when
        # the harness is unreachable, and collapsing both onto one status keeps that path single.
        raise HTTPException(status_code=404, detail={"unmeasured": True, "op_type": op_type,
                                                     "reason": r.get("reason"), "n": r.get("n", 0),
                                                     "excluded_runs": r.get("excluded_runs", 0)})
    return r
