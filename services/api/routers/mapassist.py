"""Map-assisted annotation endpoints (M3.2): map-match a session's GNSS track to the OSM road network and
fetch map-derived priors (editable hints) for a frame."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

router = APIRouter()


@router.post("/mapassist/match")
async def match(session_id: str, max_dist_m: float = 30.0):
    from services.mapassist.matcher import match_session

    return await match_session(UUID(session_id), max_dist_m)


@router.get("/mapassist/priors")
async def priors(frame_id: str):
    from services.mapassist.priors import frame_priors

    return await frame_priors(UUID(frame_id))
