"""Scoping a query to one tenant, in one place.

The corpus spine is `Session -> Frame -> Object`, and only `session` carries `project_id` (migration 0092).
That is deliberate: everything below a session reaches its tenant through `session_id`, so a scope applied
at the session is a scope on the whole spine, and 202 files that import `Object` or `Frame` do not each need
to learn about tenancy.

**This is a boundary only where it is applied.** With one tenant it is a convention, and calling it anything
stronger would be a claim the code does not support. What it does buy today is that the seam exists in one
place with a test, so making it enforceable later is a change to this file and its callers rather than an
audit of the whole tree.

The `None` case is not a hole, it is the single-tenant deployment: a caller with no project sees everything,
which is exactly what every caller does today. What matters is that asking for a scope and getting an
unscoped answer is impossible: `scope_sessions` with a project always filters, and a session with no project
is never visible to a scoped caller, because an unassigned row is not "everyone's".
"""

from __future__ import annotations

import uuid
from typing import TypeVar

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from db.models import Session as DbSession

log = get_logger("api.tenancy")

T = TypeVar("T")

DEFAULT_PROJECT_NAME = "default"


def scope_sessions(stmt: Select, project_id: str | uuid.UUID | None) -> Select:
    """Restrict a query over `session` to one project. No project means no restriction."""
    if project_id is None:
        return stmt
    return stmt.where(DbSession.project_id == uuid.UUID(str(project_id)))


def scope_frames(stmt: Select, project_id: str | uuid.UUID | None) -> Select:
    """Restrict a query over `frame` to one project, through the session that owns it."""
    if project_id is None:
        return stmt
    return stmt.where(Frame.session_id.in_(
        select(DbSession.session_id).where(DbSession.project_id == uuid.UUID(str(project_id)))))


def scope_objects(stmt: Select, project_id: str | uuid.UUID | None) -> Select:
    """Restrict a query over `object` to one project, through frame and session.

    Two hops rather than a denormalised column on `object`: the column would need backfilling across
    576,393 rows and would then need keeping true on every write, and a wrong tenant column is worse than a
    join, because it is silently wrong rather than slow.
    """
    if project_id is None:
        return stmt
    return stmt.where(Object.frame_id.in_(
        select(Frame.frame_id).where(Frame.session_id.in_(
            select(DbSession.session_id).where(DbSession.project_id == uuid.UUID(str(project_id)))))))


async def default_project_id(db: AsyncSession) -> uuid.UUID | None:
    """The project every pre-existing session was assigned to by migration 0092."""
    from db.models import LabelProject

    return (await db.execute(
        select(LabelProject.project_id).where(LabelProject.name == DEFAULT_PROJECT_NAME))).scalar_one_or_none()


async def assign_session(db: AsyncSession, session_id: str | uuid.UUID,
                         project_id: str | uuid.UUID | None = None) -> dict:
    """Put a session in a project, defaulting to the one everything else is in.

    Ingest paths that predate tenancy call this rather than leaving a null: an unassigned session is
    invisible to every scoped query, so a drive that arrives without a project would silently vanish from a
    tenant's view of their own corpus.
    """
    sid = uuid.UUID(str(session_id))
    sess = await db.get(DbSession, sid)
    if sess is None:
        raise ValueError(f"session {sid} not found")
    pid = uuid.UUID(str(project_id)) if project_id else await default_project_id(db)
    sess.project_id = pid
    await db.commit()
    log.info("tenancy.session_assigned", session=str(sid), project=str(pid) if pid else None)
    return {"session_id": str(sid), "project_id": str(pid) if pid else None}


async def unassigned_sessions(db: AsyncSession, limit: int = 100) -> list[str]:
    """Sessions with no project, which are the ones a scoped caller cannot see.

    Reported rather than swept up automatically: a session with no tenant is a question about where it came
    from, and answering it by assigning everything to the default would hide the ingest path that is not
    setting one.
    """
    # Ordered, because a LIMIT without one returns an arbitrary window that shifts as rows are inserted:
    # the same session could appear twice across two pages or never appear at all, and an operator chasing
    # a misconfigured ingest path would be reading a list that quietly changes under them.
    #
    # Newest first, since a session that arrived without a project is a question about the ingest running
    # now, not about one that ran a year ago.
    rows = (await db.execute(
        select(DbSession.session_id).where(DbSession.project_id.is_(None))
        .order_by(DbSession.created_at.desc(), DbSession.session_id)
        .limit(limit))).scalars().all()
    return [str(r) for r in rows]
