"""Multi-camera synchronized annotation endpoints (M3.1 / M-MC.0): synchronized frame groups across the rig,
group-aware navigation and confirmation, and cross-camera association into one rig identity (gated on
calibration). The `groups` read is the in-memory assembly; `groups/build` persists it into `frame_group` so
the canvas can navigate whole groups and see dropouts and out-of-tolerance windows."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/multicam/groups")
async def groups(session_id: str, tol_ms: int = 20):
    """In-memory rig groups (no writes): live assembly at an arbitrary tolerance."""
    from services.multicam.sync import frame_groups

    return await frame_groups(UUID(session_id), tol_ms * 1_000_000)


@router.post("/multicam/groups/build")
async def build_groups(session_id: str, tol_ms: int = 20):
    """Assemble and persist the session's frame groups (idempotent backfill). Returns a health summary."""
    from services.multicam.sync import persist_groups

    return await persist_groups(UUID(session_id), tol_ms * 1_000_000)


@router.get("/multicam/groups/persisted")
async def persisted_groups(session_id: str):
    """The session's persisted groups in time order, with per-group missing_cams and sync spread."""
    from services.multicam.sync import list_groups

    return await list_groups(UUID(session_id))


@router.get("/multicam/group/at")
async def group_at(session_id: str, ts_ns: int):
    """The persisted group nearest a timestamp: how the workspace opens the rig view for a given frame."""
    from services.multicam.sync import group_at_ts

    g = await group_at_ts(UUID(session_id), ts_ns)
    if g is None:
        raise HTTPException(404, "no groups for session (build them first)")
    return g


@router.get("/multicam/group/nav")
async def group_nav(session_id: str, group_id: str, direction: str = "next"):
    """The previous or next group in time (group-aware prev/next)."""
    from services.multicam.sync import adjacent_group

    if direction not in ("prev", "next"):
        raise HTTPException(400, "direction must be prev or next")
    g = await adjacent_group(UUID(session_id), UUID(group_id), direction)
    return {"group": g}


@router.post("/multicam/group/confirm")
async def group_confirm(group_id: str, confirmed: bool = True):
    """Confirm (or unconfirm) a whole rig group at once."""
    from services.multicam.sync import confirm_group

    g = await confirm_group(UUID(group_id), confirmed)
    if g is None:
        raise HTTPException(404, "group not found")
    return g


@router.post("/multicam/associate")
async def associate(session_id: str):
    from services.multicam.associate import associate_session

    return await associate_session(UUID(session_id))
