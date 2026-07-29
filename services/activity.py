"""The activity feed: what a person did, in order.

Reviews, objects, jobs, and exports each kept their own history, and none of them was a timeline. A user
could not answer "what did I do today" without reading five tables, and a lead could not see a shift's work
at all.

Recording is deliberately best-effort and never raises. An activity row is a description of work, not the
work, and a feed failure that rolled back a completed review would be a far worse bug than a missing line
in a list.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ActivityEvent, User

log = get_logger("activity")

# The verbs the feed renders. Anything else still records, but these are the ones with a label and an icon,
# so keeping the list explicit is what stops the feed becoming a wall of raw identifiers.
VERBS = {
    "signed_in": "signed in",
    "reviewed": "reviewed an object",
    "confirmed": "confirmed an object",
    "rejected": "rejected an object",
    "created_object": "drew an object",
    "deleted_object": "deleted an object",
    "submitted": "submitted for QA",
    "tagged": "tagged objects",
    "commented": "commented on an issue",
    "resolved_issue": "resolved an issue",
    "started_job": "started a job",
    "finished_job": "finished a job",
    "sealed_dataset": "sealed a dataset",
    "exported": "exported a dataset",
    "promoted_model": "promoted a model",
    "blocked_promotion": "blocked a promotion",
    "erased_session": "erased a session",
    "viewed_pii": "viewed personal data",
    "imported": "imported data",
}


async def record_activity(db: AsyncSession, *, user: User | None = None,
                          user_id: uuid.UUID | str | None = None, user_name: str | None = None,
                          verb: str, subject_type: str | None = None, subject_id: str | None = None,
                          summary: str | None = None, href: str | None = None,
                          meta: dict | None = None, commit: bool = True) -> None:
    """Append one event. Never raises: a feed failure must not roll back the work it describes."""
    try:
        uid = user.user_id if user is not None else (
            uuid.UUID(str(user_id)) if user_id else None)
        db.add(ActivityEvent(
            user_id=uid,
            user_name=(user.name if user is not None else user_name),
            verb=verb, subject_type=subject_type,
            subject_id=str(subject_id) if subject_id else None,
            summary=summary or VERBS.get(verb, verb), href=href, meta=meta or {}))
        if commit:
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("activity.record_failed", verb=verb, error=str(exc))
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def list_activity(db: AsyncSession, *, user_id: str | None = None, verb: str | None = None,
                        since_hours: int | None = None, limit: int = 100,
                        offset: int = 0) -> dict:
    """The feed, newest first. Without user_id it is the whole team's, which is what a lead reads."""
    stmt = select(ActivityEvent)
    count_stmt = select(func.count()).select_from(ActivityEvent)

    def _filters(q):
        if user_id:
            q = q.where(ActivityEvent.user_id == uuid.UUID(user_id))
        if verb:
            q = q.where(ActivityEvent.verb == verb)
        if since_hours:
            q = q.where(ActivityEvent.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
        return q

    rows = (await db.execute(
        _filters(stmt).order_by(ActivityEvent.created_at.desc())
        .offset(max(offset, 0)).limit(min(max(limit, 1), 500)))).scalars().all()
    total = (await db.execute(_filters(count_stmt))).scalar_one()
    return {"total": int(total), "offset": offset, "limit": limit,
            "events": [_dict(r) for r in rows]}


async def activity_summary(db: AsyncSession, *, user_id: str | None = None,
                           hours: int = 24) -> dict:
    """Counts per verb over a window: the "what did I do today" headline above the list."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    stmt = (select(ActivityEvent.verb, func.count())
            .where(ActivityEvent.created_at >= since).group_by(ActivityEvent.verb))
    if user_id:
        stmt = stmt.where(ActivityEvent.user_id == uuid.UUID(user_id))
    by_verb = {v: int(n) for v, n in (await db.execute(stmt)).all()}

    actors_stmt = (select(func.count(func.distinct(ActivityEvent.user_id)))
                   .where(ActivityEvent.created_at >= since))
    actors = (await db.execute(actors_stmt)).scalar_one()
    return {"hours": hours, "total": sum(by_verb.values()), "by_verb": by_verb,
            "labels": {k: VERBS.get(k, k) for k in by_verb},
            "active_people": int(actors)}


def _dict(r: ActivityEvent) -> dict:
    return {"event_id": str(r.event_id), "user_id": str(r.user_id) if r.user_id else None,
            "user_name": r.user_name, "verb": r.verb, "label": VERBS.get(r.verb, r.verb),
            "subject_type": r.subject_type, "subject_id": r.subject_id,
            "summary": r.summary, "href": r.href, "meta": r.meta or {},
            "created_at": r.created_at.isoformat() if r.created_at else None}
