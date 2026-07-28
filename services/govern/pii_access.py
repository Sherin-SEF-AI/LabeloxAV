"""The PII access log: recording that a human looked at personal data.

`PiiAudit` records what the redactor detected in a frame. Nothing recorded what anyone then looked at, so
the system could answer "what personal data does this frame contain" and not "who has seen it", which is the
half a DPDPA or GDPR enquiry actually turns on. An unredacted frame could be fetched by any authenticated
reviewer and leave no trace that it happened.

Recording is best effort and never raises. A log write that failed a frame fetch would train operators to
work around the logging, and an access record is evidence about the request, not part of serving it.

What is deliberately NOT recorded: the pixels, the plate text, or any copy of the personal data itself. A
compliance log that duplicates the data it is tracking has doubled the exposure it exists to measure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import PiiAccessLog

log = get_logger("pii_access")

ACTIONS = ("view", "download", "export", "read_plate", "listen")
SUBJECTS = ("frame", "object", "plate_read", "speech", "export", "session")


async def record_access(db: AsyncSession, *, subject_type: str, subject_id: str,
                        action: str = "view", user=None, user_id: str | None = None,
                        user_name: str | None = None, session_id: str | uuid.UUID | None = None,
                        pii_kinds: list[str] | None = None, redacted: bool = True,
                        route: str | None = None, pack_id: str | None = None,
                        commit: bool = True) -> None:
    """Record one access. Never raises: evidence about a request must not be able to fail the request."""
    try:
        uid = user.user_id if user is not None else (uuid.UUID(str(user_id)) if user_id else None)
        db.add(PiiAccessLog(
            user_id=uid, user_name=(user.name if user is not None else user_name),
            subject_type=subject_type, subject_id=str(subject_id),
            session_id=uuid.UUID(str(session_id)) if session_id else None,
            action=action, pii_kinds=list(pii_kinds or []), redacted=bool(redacted),
            route=route, pack_id=pack_id))
        if commit:
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("pii_access.record_failed", subject=str(subject_id), error=str(exc))
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def list_access(db: AsyncSession, *, user_id: str | None = None, subject_id: str | None = None,
                      session_id: str | None = None, action: str | None = None,
                      since_hours: int | None = None, limit: int = 100, offset: int = 0) -> dict:
    """The enquiry query: filterable by who, by what, and by when."""
    stmt = select(PiiAccessLog)
    count_stmt = select(func.count()).select_from(PiiAccessLog)

    def _filters(q):
        if user_id:
            q = q.where(PiiAccessLog.user_id == uuid.UUID(user_id))
        if subject_id:
            q = q.where(PiiAccessLog.subject_id == str(subject_id))
        if session_id:
            q = q.where(PiiAccessLog.session_id == uuid.UUID(session_id))
        if action:
            q = q.where(PiiAccessLog.action == action)
        if since_hours:
            q = q.where(PiiAccessLog.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
        return q

    rows = (await db.execute(
        _filters(stmt).order_by(PiiAccessLog.created_at.desc())
        .offset(max(offset, 0)).limit(min(max(limit, 1), 500)))).scalars().all()
    total = (await db.execute(_filters(count_stmt))).scalar_one()
    return {"total": int(total), "offset": offset, "limit": limit,
            "accesses": [_dict(r) for r in rows]}


async def access_summary(db: AsyncSession, *, hours: int = 168) -> dict:
    """Headline counts over a window, including how much was served unredacted.

    Unredacted access is broken out because it is the number that matters: viewing a blurred frame is
    ordinary work, and viewing the original is the thing a policy is written about.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)
    base = select(func.count()).select_from(PiiAccessLog).where(PiiAccessLog.created_at >= since)
    total = (await db.execute(base)).scalar_one()
    unredacted = (await db.execute(
        base.where(PiiAccessLog.redacted.is_(False)))).scalar_one()

    by_action = {a: int(n) for a, n in (await db.execute(
        select(PiiAccessLog.action, func.count())
        .where(PiiAccessLog.created_at >= since)
        .group_by(PiiAccessLog.action))).all()}
    by_user = {(u or "unknown"): int(n) for u, n in (await db.execute(
        select(PiiAccessLog.user_name, func.count())
        .where(PiiAccessLog.created_at >= since)
        .group_by(PiiAccessLog.user_name)
        .order_by(func.count().desc()).limit(20))).all()}
    return {"hours": hours, "accesses": int(total), "unredacted": int(unredacted),
            "by_action": by_action, "by_user": by_user}


async def subject_history(db: AsyncSession, subject_id: str) -> list[dict]:
    """Everything ever recorded about one subject: the answer handed to a data-subject request."""
    rows = (await db.execute(
        select(PiiAccessLog).where(PiiAccessLog.subject_id == str(subject_id))
        .order_by(PiiAccessLog.created_at.desc()))).scalars().all()
    return [_dict(r) for r in rows]


def _dict(r: PiiAccessLog) -> dict:
    return {"access_id": str(r.access_id), "user_id": str(r.user_id) if r.user_id else None,
            "user_name": r.user_name, "subject_type": r.subject_type, "subject_id": r.subject_id,
            "session_id": str(r.session_id) if r.session_id else None,
            "action": r.action, "pii_kinds": r.pii_kinds or [], "redacted": r.redacted,
            "route": r.route, "pack_id": r.pack_id,
            "created_at": r.created_at.isoformat() if r.created_at else None}
