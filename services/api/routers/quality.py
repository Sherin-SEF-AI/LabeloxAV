"""Gate B (M9) endpoints: the quality sheet, gold-set listing, and the seal/fit triggers.

The sheet is served from cached metrics (no GPU in the request). Measuring is done out of band via
`make m9`. Sealing a gold set and fitting calibration are CPU/IO and run inline here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.analytics.quality import list_gold_sets, quality_sheet
from services.api.deps import CalibrateFitIn, GoldSealIn
from services.autolabel.isotonic import fit_isotonic
from services.training.gold import GoldSpec, seal_gold

router = APIRouter()


@router.get("/quality/gold-sets")
async def gold_sets():
    return await list_gold_sets()


@router.get("/quality/sheet")
async def sheet(gold_id: str):
    res = await quality_sheet(gold_id)
    if not res.get("found"):
        raise HTTPException(status_code=404, detail=f"gold set {gold_id} not found")
    return res


@router.post("/quality/gold/seal")
async def seal(payload: GoldSealIn):
    spec = GoldSpec(name=payload.name, cities=payload.cities, session_id=payload.session_id,
                    class_names=payload.class_names, limit=payload.limit)
    try:
        return await seal_gold(spec)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/quality/calibrate/fit")
async def calibrate_fit(payload: CalibrateFitIn):
    try:
        return await fit_isotonic(payload.gold_id, payload.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
