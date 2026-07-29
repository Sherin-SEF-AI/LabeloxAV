"""The inbox: notifications, the activity feed, and the PII access log.

Three read surfaces that all answer "what happened", at different distances. Notifications are what somebody
needs to act on, activity is what people did, and the access log is what a regulator asks for.

The access log is admin-only. It records who looked at personal data, which makes it a record about
employees as much as about subjects, and a reviewer being able to browse their colleagues' viewing history
is surveillance rather than compliance.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role, require_user

router = APIRouter()


# ---------------------------------------------------------------- notifications

@router.get("/notifications")
async def list_notifications(unread_only: bool = False, limit: int = Query(50, ge=1, le=200),
                             offset: int = Query(0, ge=0), user=Depends(require_user),
                             db: AsyncSession = Depends(db_session)):
    from services import notify

    return await notify.list_for(db, user, unread_only=unread_only, limit=limit, offset=offset)


@router.get("/notifications/count")
async def count_notifications(user=Depends(require_user), db: AsyncSession = Depends(db_session)):
    """What the bell renders. Cheap enough to call on every page mount."""
    from services import notify

    return {"unread": await notify.unread_count(db, user)}


@router.post("/notifications/{notification_id}/read")
async def read_notification(notification_id: str, user=Depends(require_user),
                            db: AsyncSession = Depends(db_session)):
    from services import notify

    return await notify.mark_read(db, user, notification_id)


@router.post("/notifications/read-all")
async def read_all(user=Depends(require_user), db: AsyncSession = Depends(db_session)):
    from services import notify

    return await notify.mark_all_read(db, user)


# ---------------------------------------------------------------- activity

@router.get("/activity")
async def list_activity(user_id: str | None = None, verb: str | None = None,
                        since_hours: int | None = None, mine: bool = False,
                        limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                        user=Depends(require_user), db: AsyncSession = Depends(db_session)):
    """The feed. `mine=true` is the "what did I do today" view; without it, the whole team's, which is what
    a lead reads."""
    from services.activity import list_activity as _list

    return await _list(db, user_id=(str(user.user_id) if mine else user_id), verb=verb,
                       since_hours=since_hours, limit=limit, offset=offset)


@router.get("/activity/summary")
async def activity_summary(hours: int = Query(24, ge=1, le=720), mine: bool = True,
                           user=Depends(require_user), db: AsyncSession = Depends(db_session)):
    from services.activity import activity_summary as _summary

    return await _summary(db, user_id=(str(user.user_id) if mine else None), hours=hours)


# ---------------------------------------------------------------- PII access log

@router.get("/govern/pii-access", dependencies=[Depends(require_role("admin"))])
async def pii_access(user_id: str | None = None, subject_id: str | None = None,
                     session_id: str | None = None, action: str | None = None,
                     since_hours: int | None = None,
                     limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                     db: AsyncSession = Depends(db_session)):
    """Who viewed personal data. Admin-only: it is a record about employees as much as about subjects."""
    from services.govern.pii_access import list_access

    return await list_access(db, user_id=user_id, subject_id=subject_id, session_id=session_id,
                             action=action, since_hours=since_hours, limit=limit, offset=offset)


@router.get("/govern/pii-access/summary", dependencies=[Depends(require_role("admin"))])
async def pii_access_summary(hours: int = Query(168, ge=1, le=8760),
                             db: AsyncSession = Depends(db_session)):
    from services.govern.pii_access import access_summary

    return await access_summary(db, hours=hours)
