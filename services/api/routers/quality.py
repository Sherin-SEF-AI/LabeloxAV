"""Gate B (M9) endpoints: the quality sheet, gold-set listing, and the seal/fit triggers.

The sheet is served from cached metrics (no GPU in the request). Measuring is done out of band via
`make m9`. Sealing a gold set and fitting calibration are CPU/IO and run inline here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.analytics.quality import list_gold_sets, quality_sheet
from services.api.deps import CalibrateFitIn, GoldSealIn
from services.autolabel.isotonic import fit_isotonic
from services.training.gold import GoldSpec, seal_gold

router = APIRouter()


class IaaIn(BaseModel):
    set_a: list                     # [{bbox:[x1,y1,x2,y2], class_name}]
    set_b: list
    iou_thresh: float = 0.5


@router.post("/quality/iaa")
async def inter_annotator_agreement(body: IaaIn):
    """Milestone I: inter-annotator agreement between two independent label sets on a frame: detection
    agreement, class agreement, mean IoU, and Cohen's kappa."""
    from services.quality.iaa import iaa_score
    return iaa_score(body.set_a, body.set_b, body.iou_thresh)


@router.get("/quality/attr-audit/{session_id}")
async def attr_audit(session_id: str):
    """Milestone I: scan a session's objects for attribute-schema violations against each object's current
    class (surfaces labels invalidated by a class change or written before validation)."""
    from uuid import UUID

    from services.quality.attr_audit import session_attr_audit
    return await session_attr_audit(UUID(session_id))


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
