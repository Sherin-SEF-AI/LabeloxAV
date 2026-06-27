"""Multi-camera synchronized annotation endpoints (M3.1): synchronized frame groups across the rig, and
cross-camera association into one rig identity (gated on calibration)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

router = APIRouter()


@router.get("/multicam/groups")
async def groups(session_id: str, tol_ms: int = 20):
    from services.multicam.sync import frame_groups

    return await frame_groups(UUID(session_id), tol_ms * 1_000_000)


@router.post("/multicam/associate")
async def associate(session_id: str):
    from services.multicam.associate import associate_session

    return await associate_session(UUID(session_id))
