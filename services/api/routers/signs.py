"""Traffic-sign recognition endpoints (M2.3): the Indian RTO taxonomy + SigLIP 2 zero-shot typing."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

router = APIRouter()


@router.get("/signs/taxonomy")
async def taxonomy():
    from services.autolabel.signs.taxonomy import get_sign_taxonomy

    t = get_sign_taxonomy()
    return {"version": t["version"], "categories": t["categories"], "types": t["types"]}


@router.post("/signs/recognize")
async def recognize(session_id: str, limit: int | None = None):
    """Type the session's traffic_sign detections against the taxonomy; flag text-bearing for OCR."""
    from services.autolabel.signs.recognize import recognize_session

    return await recognize_session(UUID(session_id), limit)
