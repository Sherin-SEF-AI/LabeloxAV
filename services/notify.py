"""Notifications: telling somebody that something happened.

The system already knew when an issue was raised, a job finished, a promotion was blocked, drift breached,
or an SLO alarmed. None of it reached a person unless they happened to be looking at the right page at the
right moment, which meant a blocked promotion could sit unnoticed for a day and an assigned job could sit
unnoticed for longer.

Two design points:

- **A notification is addressed to a user or to a role, never broadcast.** Role addressing is what makes
  "whoever is on duty" expressible without inventing a group model, and read state for those lives in a
  join table because there is no single owner to mark.
- **A second event about the same subject supersedes the first.** Drift and SLO breaches re-evaluate on a
  schedule; without superseding, an unresolved condition would produce a new line every cycle and the bell
  would become noise nobody reads, which is the same as having no bell.

Emission never raises. A notification is a message about work, not the work, and failing a promotion
because its announcement failed would be a far worse bug than a missing line in a list.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Notification, NotificationRead, User

log = get_logger("notify")

SEVERITIES = ("info", "warn", "critical")

# Every kind the app emits, with the audience that should see it by default. Explicit so a new emitter has
# to state who it is for rather than defaulting to everyone.
KINDS: dict[str, str] = {
    "issue_opened": "reviewer",
    "issue_comment": "reviewer",
    "job_assigned": "annotator",
    "job_finished": "annotator",
    "job_failed": "annotator",
    "promotion_blocked": "reviewer",
    "model_promoted": "reviewer",
    "kill_switch": "admin",
    "drift_breach": "reviewer",
    "slo_breach": "admin",
    "gate_batch_ready": "reviewer",
    "watchlist_hit": "reviewer",
    "export_ready": "reviewer",
    "erasure_complete": "admin",
}


async def notify(db: AsyncSession, *, kind: str, title: str, body: str | None = None,
                 user_id: uuid.UUID | str | None = None, role: str | None = None,
                 severity: str = "info", href: str | None = None,
                 subject_type: str | None = None, subject_id: str | None = None,
                 meta: dict | None = None, supersede: bool = True,
                 commit: bool = True) -> str | None:
    """Emit one notification. Returns its id, or None if it could not be recorded.

    With `supersede`, an unread notification of the same kind about the same subject is replaced rather than
    added to, so a condition that re-evaluates on a schedule stays one line.
    """
    try:
        if severity not in SEVERITIES:
            severity = "info"
        if user_id is None and role is None:
            role = KINDS.get(kind, "reviewer")

        if supersede and subject_type and subject_id:
            existing = (await db.execute(
                select(Notification).where(
                    Notification.kind == kind,
                    Notification.subject_type == subject_type,
                    Notification.subject_id == str(subject_id),
                    Notification.read_at.is_(None)))).scalars().all()
            for old in existing:
                await db.delete(old)

        row = Notification(
            user_id=uuid.UUID(str(user_id)) if user_id else None,
            role=role, kind=kind, severity=severity, title=title, body=body, href=href,
            subject_type=subject_type, subject_id=str(subject_id) if subject_id else None,
            meta=meta or {})
        db.add(row)
        if commit:
            await db.commit()
            await db.refresh(row)
        else:
            await db.flush()
        _publish(row)
        return str(row.notification_id)
    except Exception as exc:  # noqa: BLE001
        # Swallowed on purpose: a failed notification must not roll back the work it was announcing. Captured
        # so the failure is not only a warning line that nobody reads, which is how a notification channel
        # goes quietly dead.
        from core.observability import capture

        capture(exc, where="notify", kind=kind, subject_type=subject_type, subject_id=subject_id)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def _publish(row: Notification) -> None:
    """Wake the SSE stream so the bell updates without a poll. Best effort: the stream also re-reads on its
    own interval, so a missed wake costs latency, never a lost notification."""
    try:
        from services.api.routers.events import publish

        publish("notifications", {"kind": row.kind, "severity": row.severity,
                                  "user_id": str(row.user_id) if row.user_id else None,
                                  "role": row.role})
    except Exception:  # noqa: BLE001
        pass


def _audience(user: User):
    """The rows this user should see: addressed to them, or to their role."""
    return or_(Notification.user_id == user.user_id, Notification.role == user.role)


async def list_for(db: AsyncSession, user: User, *, unread_only: bool = False,
                   limit: int = 50, offset: int = 0) -> dict:
    read_ids = set()
    if True:
        rows = (await db.execute(
            select(NotificationRead.notification_id)
            .where(NotificationRead.user_id == user.user_id))).scalars().all()
        read_ids = {str(r) for r in rows}

    stmt = select(Notification).where(_audience(user))
    if unread_only:
        # A role-addressed row has no read_at of its own, so its read state lives in the join table and is
        # filtered below rather than in SQL.
        stmt = stmt.where(or_(Notification.read_at.is_(None), Notification.user_id.is_(None)))
    rows = (await db.execute(
        stmt.order_by(Notification.created_at.desc())
        .offset(max(offset, 0)).limit(min(max(limit, 1), 200)))).scalars().all()

    out = []
    for r in rows:
        read = bool(r.read_at) if r.user_id else (str(r.notification_id) in read_ids)
        if unread_only and read:
            continue
        out.append({**_dict(r), "read": read})
    total = (await db.execute(
        select(func.count()).select_from(Notification).where(_audience(user)))).scalar_one()
    return {"total": int(total), "offset": offset, "limit": limit, "notifications": out}


async def unread_count(db: AsyncSession, user: User) -> int:
    """What the bell shows. Two queries rather than one so the role-addressed half stays correct."""
    personal = (await db.execute(
        select(func.count()).select_from(Notification)
        .where(Notification.user_id == user.user_id, Notification.read_at.is_(None)))).scalar_one()

    role_ids = (await db.execute(
        select(Notification.notification_id).where(Notification.role == user.role))).scalars().all()
    if not role_ids:
        return int(personal)
    seen = (await db.execute(
        select(func.count()).select_from(NotificationRead)
        .where(NotificationRead.user_id == user.user_id,
               NotificationRead.notification_id.in_(role_ids)))).scalar_one()
    return int(personal) + max(0, len(role_ids) - int(seen))


async def mark_read(db: AsyncSession, user: User, notification_id: str) -> dict:
    row = await db.get(Notification, uuid.UUID(notification_id))
    if row is None:
        return {"marked": False, "reason": "not found"}
    if row.user_id == user.user_id:
        row.read_at = row.read_at or datetime.now(UTC)
    else:
        if row.role != user.role:
            return {"marked": False, "reason": "not addressed to you"}
        exists = await db.get(NotificationRead, (row.notification_id, user.user_id))
        if exists is None:
            db.add(NotificationRead(notification_id=row.notification_id, user_id=user.user_id))
    await db.commit()
    return {"marked": True, "notification_id": notification_id}


async def mark_all_read(db: AsyncSession, user: User) -> dict:
    rows = (await db.execute(select(Notification).where(_audience(user)))).scalars().all()
    now = datetime.now(UTC)
    n = 0
    for r in rows:
        if r.user_id == user.user_id:
            if r.read_at is None:
                r.read_at = now
                n += 1
        elif await db.get(NotificationRead, (r.notification_id, user.user_id)) is None:
            db.add(NotificationRead(notification_id=r.notification_id, user_id=user.user_id))
            n += 1
    await db.commit()
    return {"marked": n}


async def prune(db: AsyncSession, *, older_than_days: int = 90) -> dict:
    """Drop read notifications past a horizon, so the table does not grow without bound."""
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    rows = (await db.execute(
        select(Notification).where(Notification.created_at < cutoff,
                                   Notification.read_at.isnot(None)))).scalars().all()
    for r in rows:
        await db.delete(r)
    await db.commit()
    return {"pruned": len(rows)}


def _dict(r: Notification) -> dict:
    return {"notification_id": str(r.notification_id), "kind": r.kind, "severity": r.severity,
            "title": r.title, "body": r.body, "href": r.href,
            "subject_type": r.subject_type, "subject_id": r.subject_id,
            "role": r.role, "user_id": str(r.user_id) if r.user_id else None,
            "meta": r.meta or {},
            "created_at": r.created_at.isoformat() if r.created_at else None}
