"""Who may see which sessions, decided in one place.

A project is "team" (everyone signed in sees its sessions, which is every project unless someone changes it)
or "members" (its members and administrators only). The rule is phrased as what a person may NOT see, because
that keeps the common case free: when no project is members-only, or the caller is an administrator, nothing
is hidden and no query changes.

Enforced at three layers (see services/api/main.py):
  1. by id: any route with a session, frame, object or track id in its path or query answers 404 for an item
     in a project hidden from the caller. 404 rather than 403, so the response does not confirm it exists.
  2. in the listings a person browses, scoped in the query so totals and pages stay right: sessions, the
     home page, the command palette's search, the review and triage queues.
  3. under every other JSON response, a net that drops rows naming a hidden item (services/api/project_filter.py).
Aggregates (analytics charts and the like) still count across projects; they name no item.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, LabelProject, Object, ProjectMember, Track
from db.models import Session as DbSession

TEAM, MEMBERS = "team", "members"
VISIBILITIES = (TEAM, MEMBERS)


def _uid(user) -> uuid.UUID | None:
    raw = getattr(user, "user_id", None)
    try:
        return uuid.UUID(str(raw)) if raw else None
    except ValueError:
        return None


async def hidden_projects(db: AsyncSession, user) -> frozenset[uuid.UUID]:
    """The projects whose sessions this person may not see. Empty for an administrator, for anyone when no
    project is members-only, and for an unauthenticated dev server (auth off, no user)."""
    if user is None or getattr(user, "role", None) == "admin":
        return frozenset()
    restricted = select(LabelProject.project_id).where(LabelProject.visibility == MEMBERS)
    uid = _uid(user)
    if uid is not None:
        restricted = restricted.where(LabelProject.project_id.not_in(
            select(ProjectMember.project_id).where(ProjectMember.user_id == uid)))
    return frozenset((await db.execute(restricted)).scalars().all())


def as_hidden(hidden: object) -> frozenset[uuid.UUID]:
    """The hidden set a route was handed, or nothing hidden when it was called as a plain function (a service
    or a test calling the route directly passes no caller, so its dependency default arrives unresolved)."""
    return hidden if isinstance(hidden, frozenset) else frozenset()


def scope_visible_sessions(stmt: Select, hidden: frozenset[uuid.UUID]) -> Select:
    """Restrict a query over `session` to what the caller may see. A session with no project is visible: it
    belongs to no members-only project (and migration 0118 put every such session in the default one)."""
    hidden = as_hidden(hidden)
    if not hidden:
        return stmt
    return stmt.where(or_(DbSession.project_id.is_(None), DbSession.project_id.not_in(hidden)))


def visible_session_ids(hidden: frozenset[uuid.UUID]):
    """A subquery of the session ids the caller may see, for filtering frames, objects and tracks."""
    return select(DbSession.session_id).where(
        or_(DbSession.project_id.is_(None), DbSession.project_id.not_in(hidden)))


async def project_of(db: AsyncSession, kind: str, raw_id: str) -> tuple[bool, uuid.UUID | None]:
    """The project an item belongs to, by the kind of id a route carries. Returns (found, project_id).

    Not found is reported as found=False so the caller lets the route answer its own 404; only an item that
    exists in a hidden project is refused here.
    """
    try:
        ident = uuid.UUID(str(raw_id))
    except ValueError:
        return False, None
    if kind == "session_id":
        stmt = select(DbSession.project_id).where(DbSession.session_id == ident)
    elif kind == "frame_id":
        stmt = select(DbSession.project_id).join(Frame, Frame.session_id == DbSession.session_id) \
            .where(Frame.frame_id == ident)
    elif kind == "object_id":
        stmt = select(DbSession.project_id).join(Frame, Frame.session_id == DbSession.session_id) \
            .join(Object, Object.frame_id == Frame.frame_id).where(Object.object_id == ident)
    elif kind == "track_id":
        stmt = select(DbSession.project_id).join(Track, Track.session_id == DbSession.session_id) \
            .where(Track.track_id == ident)
    else:
        return False, None
    row = (await db.execute(stmt)).first()
    return (row is not None), (row[0] if row else None)


GUARDED_PARAMS = ("session_id", "frame_id", "object_id", "track_id")


async def drop_hidden(db: AsyncSession, hidden: frozenset[uuid.UUID], rows: list[dict],
                      key: str = "object_id") -> list[dict]:
    """The rows of a listing whose item the caller may see, resolving every id in one query.

    For listings built deep inside a service, where scoping the query itself would mean threading the caller
    through code that has no notion of one. `key` names the field carrying a session, frame or object id. A
    row whose id cannot be resolved is kept: it names nothing in a hidden project.
    """
    hidden = as_hidden(hidden)
    if not hidden or not rows:
        return rows
    ids: set[uuid.UUID] = set()
    for r in rows:
        try:
            ids.add(uuid.UUID(str(r.get(key))))
        except (TypeError, ValueError):
            continue
    if not ids:
        return rows
    if key == "session_id":
        stmt = select(DbSession.session_id, DbSession.project_id).where(DbSession.session_id.in_(ids))
    elif key == "frame_id":
        stmt = select(Frame.frame_id, DbSession.project_id).join(
            DbSession, DbSession.session_id == Frame.session_id).where(Frame.frame_id.in_(ids))
    elif key == "object_id":
        stmt = select(Object.object_id, DbSession.project_id).join(Frame, Frame.frame_id == Object.frame_id) \
            .join(DbSession, DbSession.session_id == Frame.session_id).where(Object.object_id.in_(ids))
    else:
        raise ValueError(f"cannot scope rows by {key!r}")
    blocked = {str(i) for i, pid in (await db.execute(stmt)).all() if pid in hidden}
    return [r for r in rows if str(r.get(key)) not in blocked]


async def first_hidden_item(db: AsyncSession, hidden: frozenset[uuid.UUID], *params: dict) -> str | None:
    """The first path or query parameter naming an item in a hidden project, or None when the request may
    proceed. Query parameters count too: many routes take ?session_id= rather than a path segment."""
    if not hidden:
        return None
    for source in params:
        for name in GUARDED_PARAMS:
            if name in source and source[name]:
                found, pid = await project_of(db, name, source[name])
                if found and pid is not None and pid in hidden:
                    return name
    return None


__all__ = ["GUARDED_PARAMS", "MEMBERS", "TEAM", "VISIBILITIES", "as_hidden", "drop_hidden", "first_hidden_item", "hidden_projects",
           "project_of", "scope_visible_sessions", "visible_session_ids"]
