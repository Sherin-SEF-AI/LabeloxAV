"""Presence and the frame edit lock, over HTTP.

Polling, on purpose. The heartbeat an editor already has to send to stay present carries back everything it
needs to know (who else is here, who holds the frame, whether that is still this tab), so there is no
second channel to keep in step with the first and nothing to reconnect.

Every route names the frame in its path, which is what puts it under the project guard this router is
mounted with: a frame in a members-only project answers 404 to anyone outside it, here as everywhere else,
so presence cannot be used to learn who is working in a project the caller cannot see.

Who the caller is comes from their credential and from nowhere else. The body carries one thing, the tab's
own token, and that token only ever says "the same tab as before"; it is never an identity.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from services.activity import record_activity
from services.api.deps import current_user, db_session, role_rank
from services.collab.presence import (
    TOKEN_RE,
    TakeOverDenied,
    degraded_state,
    get_presence,
    identity_of,
)

router = APIRouter(prefix="/presence")

# Who may take a frame from somebody else. A reviewer already outranks an annotator's work everywhere else
# in the review flow; this is the same floor.
TAKE_OVER_ROLE = "reviewer"


class TabIn(BaseModel):
    # Random, minted by the tab, and never shown to another tab. Its alphabet is closed because the store
    # splices it into a value with a separator.
    tab: str = Field(..., pattern=TOKEN_RE.pattern)


class HeartbeatIn(TabIn):
    # Take the lock too, if nobody holds it. An editor sends true; a tab that has been told somebody else
    # is editing sends false and stays a viewer until its user asks otherwise.
    acquire: bool = False


class ReleaseIn(TabIn):
    # Stop being present as well. False keeps the tab listed, for an editor stepping back to viewing.
    leave: bool = True


async def _caller(user=Depends(current_user)):
    """The authenticated user, or None on a dev server running with auth off and no identity chosen.

    With auth on there is no anonymous presence: the middleware has already required a token, and a token
    revoked since then is a 401 like anywhere else. With auth off there may be nobody to name, and the
    routes then answer the same "degraded" state they give when Redis is down, so the editor still opens.
    """
    if user is None and get_settings().auth.enabled:
        raise HTTPException(status_code=401, detail="authentication required (Bearer token)")
    return user


def _privileged(user) -> bool:
    return role_rank(user.role) >= role_rank(TAKE_OVER_ROLE)


@router.post("/frames/{frame_id}/heartbeat")
async def heartbeat(frame_id: UUID, payload: HeartbeatIn, user=Depends(_caller)):
    """This tab is on the frame. Returns who else is, and who holds the edit lock."""
    if user is None:
        return degraded_state(str(frame_id), "no signed-in user")
    return await get_presence().heartbeat(str(frame_id), identity_of(user), payload.tab,
                                          acquire=payload.acquire, privileged=_privileged(user))


@router.post("/frames/{frame_id}/lock/acquire")
async def acquire_lock(frame_id: UUID, payload: TabIn, user=Depends(_caller)):
    """Take the edit lock if it is free. Not an error when it is not: the answer says who holds it."""
    if user is None:
        return degraded_state(str(frame_id), "no signed-in user")
    return await get_presence().acquire(str(frame_id), identity_of(user), payload.tab,
                                        privileged=_privileged(user))


@router.post("/frames/{frame_id}/lock/release")
async def release_lock(frame_id: UUID, payload: ReleaseIn, user=Depends(_caller)):
    """Give the lock back. Only the tab that holds it can; anyone else's call releases nothing."""
    if user is None:
        return {"frame_id": str(frame_id), "released": False, "degraded": True}
    return await get_presence().release(str(frame_id), identity_of(user), payload.tab, leave=payload.leave)


@router.post("/frames/{frame_id}/lock/takeover")
async def take_over_lock(frame_id: UUID, payload: TabIn, user=Depends(_caller),
                         db: AsyncSession = Depends(db_session)):
    """Take the frame from whoever is editing it. Reviewers and admins; anyone, from their own other tab.

    Recorded in the activity feed whenever it displaces a different person, because the person displaced
    was mid-edit and is entitled to know who interrupted them.
    """
    if user is None:
        return degraded_state(str(frame_id), "no signed-in user")
    me = identity_of(user)
    try:
        state, displaced = await get_presence().take_over(str(frame_id), me, payload.tab,
                                                          privileged=_privileged(user))
    except TakeOverDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if displaced is not None and displaced.user_id != me.user_id:
        await record_activity(
            db, user=user, verb="took_over_frame", subject_type="frame", subject_id=str(frame_id),
            summary=f"took over editing a frame from {displaced.name}", href=f"/frame/{frame_id}",
            meta={"previous_holder_id": displaced.user_id, "previous_holder_name": displaced.name,
                  "previous_holder_role": displaced.role, "held_since_ms": displaced.since_ms})
    return state
