"""Hardening + scale endpoints (M19): the per-plane SLO observability board and ticker, byte-stable
reproducibility check, and the label-budget efficiency report. Mounted under /api."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from services.hardening.efficiency import efficiency_report
from services.hardening.repro import check_reproducible
from services.hardening.run import record_slo, slo_board

router = APIRouter()


class SloIn(BaseModel):
    plane: str
    measurements: dict
    window_s: float = 0.0


@router.post("/hardening/slo")
async def slo_tick(payload: SloIn, db: AsyncSession = Depends(db_session)):
    """Evaluate a plane's measured metrics against its SLO and record the tick to the observability ledger."""
    return await record_slo(db, payload.plane, payload.measurements, payload.window_s)


@router.get("/hardening/slo/board")
async def slo_board_view(db: AsyncSession = Depends(db_session)):
    """The latest SLO evaluation per plane, and whether every plane is meeting its SLO."""
    return await slo_board(db)


class ReproIn(BaseModel):
    run_a: dict
    run_b: dict


@router.post("/hardening/reproducible")
async def reproducible(payload: ReproIn):
    """Check two build outputs for byte-stable reproducibility, naming the first diverging field if they
    differ."""
    return check_reproducible(payload.run_a, payload.run_b)


class EfficiencyIn(BaseModel):
    entries: list[dict]              # [{slice, labels, map_before, map_after}]


@router.post("/hardening/efficiency")
async def efficiency(payload: EfficiencyIn):
    """The label-budget efficiency report: per-slice ROI ranked best-first, the overall gain per 1000 labels,
    and the slices whose spend returned nothing."""
    return efficiency_report(payload.entries)
