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
from services.training.op_precision import OPERATION_KINDS, measure_all, measure_operation

router = APIRouter()


@router.get("/eval/operations", dependencies=[Depends(require_role("annotator"))])
async def operations(db: AsyncSession = Depends(db_session)) -> dict:
    """Every operation kind with its measurement state, for the panel header."""
    return {"operations": await measure_all(db), "kinds": list(OPERATION_KINDS)}


@router.get("/eval/operations/{op_type}/latest", dependencies=[Depends(require_role("annotator"))])
async def operation_latest(op_type: str, db: AsyncSession = Depends(db_session)) -> dict:
    """The latest measurement for one operation, or 404 when there is not enough evidence to have one."""
    if op_type not in OPERATION_KINDS:
        raise HTTPException(status_code=404, detail=f"unknown operation type {op_type!r}")
    r = await measure_operation(db, op_type)
    if not r.get("measured"):
        # 404 rather than a 200 carrying nulls: the client's unmeasured path is the same one it uses when
        # the harness is unreachable, and collapsing both onto one status keeps that path single.
        raise HTTPException(status_code=404, detail={"unmeasured": True, "op_type": op_type,
                                                     "reason": r.get("reason"), "n": r.get("n", 0)})
    return r
