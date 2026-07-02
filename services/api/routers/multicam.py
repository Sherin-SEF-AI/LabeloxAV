"""Multi-camera synchronized annotation endpoints (M3.1 / M-MC.0): synchronized frame groups across the rig,
group-aware navigation and confirmation, and cross-camera association into one rig identity (gated on
calibration). The `groups` read is the in-memory assembly; `groups/build` persists it into `frame_group` so
the canvas can navigate whole groups and see dropouts and out-of-tolerance windows."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


# M-MC.2 rig identity + linked selection (Tier 1, no calibration required)


@router.get("/multicam/rig-objects")
async def rig_objects_ep(session_id: str, group_id: str):
    """The rig-first object list for a group: linked identities + unlinked singletons."""
    from services.multicam.rigident import rig_objects

    return await rig_objects(UUID(session_id), UUID(group_id))


@router.get("/multicam/suggest-links")
async def suggest_links_ep(session_id: str, group_id: str, appearance_cos: float = 0.55):
    """DINOv3 appearance-based cross-camera link candidates (assist only, never applied)."""
    from services.multicam.rigident import suggest_links

    return await suggest_links(UUID(session_id), UUID(group_id), appearance_cos)


class LinkBody(BaseModel):
    session_id: str
    group_id: str
    object_ids: list[str]
    source: str = "manual"


@router.post("/multicam/link")
async def link_ep(body: LinkBody):
    """Bind two or more objects across views into one rig identity (manual or accepted appearance suggestion)."""
    from services.multicam.rigident import link_objects

    res = await link_objects(UUID(body.session_id), UUID(body.group_id),
                             [UUID(o) for o in body.object_ids], body.source)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.post("/multicam/unlink")
async def unlink_ep(object_id: str):
    """Remove an object from its rig identity (dissolves the identity if fewer than two members remain)."""
    from services.multicam.rigident import unlink_object

    res = await unlink_object(UUID(object_id))
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


# M-MC.3 annotate-once propagate (Tier 2, calibration-gated)


@router.post("/multicam/propagate")
async def propagate_ep(object_id: str, use_sam: bool = True):
    """Place a source object into the other rig views by lens-aware projection (Tier 2). Returns gated=True
    when the session is not calibrated, so the client shows the Tier 1 (manual link) chip instead."""
    from services.multicam.propagate import propagate_object

    res = await propagate_object(UUID(object_id), use_sam)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


# M-MC.4 cross-view track handoff + consistency


@router.post("/multicam/rig-tracks/build")
async def rig_tracks_build_ep(session_id: str):
    """Chain the per-instant rig identities into rig tracks across time and cameras (via per-camera tracks)."""
    from services.multicam.rigtrack import build_rig_tracks

    return await build_rig_tracks(UUID(session_id))


@router.get("/multicam/rig-tracks")
async def rig_tracks_ep(session_id: str):
    """The session's rig tracks: instants, cameras, time span, voted class, and inconsistency flag."""
    from services.multicam.rigtrack import rig_tracks

    return await rig_tracks(UUID(session_id))


@router.get("/multicam/rig-track/timeline")
async def rig_track_timeline_ep(session_id: str, rig_track_id: str):
    """One rig track's ordered instants (the cross-camera handoff over time)."""
    from services.multicam.rigtrack import rig_track_timeline

    return await rig_track_timeline(UUID(session_id), UUID(rig_track_id))


@router.post("/multicam/consistency-check")
async def consistency_check_ep(session_id: str):
    """Flag cross-view class disagreement on rig tracks as cross_cam_inconsistent error candidates."""
    from services.multicam.rigtrack import check_consistency

    return await check_consistency(UUID(session_id))
