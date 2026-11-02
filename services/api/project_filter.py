"""The listing half of project visibility, as a safety net under every JSON response.

The by-id guard (services/api/project_guard.py) refuses a request that NAMES a hidden item. Listings are the
other way in: explore, search, datasets, scenarios and the rest return rows that carry a session, frame,
object or track id, and scoping each of those queries by hand is a change to dozens of routes that the next
new route would forget. So after a route answers, any row in a JSON list that names an item in a project
hidden from the caller is dropped.

Routes that already scope their own queries (sessions, home, find, the review queues; see
services/project_access.py) still do, which keeps their totals and pagination right; this layer only catches
what they do not. It does nothing, and reads no body, when nothing is hidden from the caller: an
administrator, a single-team deployment, or any team with no members-only project.

A filtered list can come back shorter than the limit asked for, and a total beside it may count rows the
caller no longer receives. That is the price of a net: a short page, rather than another team's data.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy import select
from starlette.responses import Response

from db.models import Frame, Object, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.project_access import as_hidden

ID_KEYS = ("session_id", "frame_id", "object_id", "track_id")
MAX_BODY = 32 * 1024 * 1024  # a larger body is a download, not a listing; it streams through untouched


def _collect(node: Any, out: dict[str, set[uuid.UUID]]) -> None:
    if isinstance(node, dict):
        for k in ID_KEYS:
            v = node.get(k)
            if isinstance(v, str):
                try:
                    out[k].add(uuid.UUID(v))
                except ValueError:
                    pass
        for v in node.values():
            _collect(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect(v, out)


async def _blocked(ids: dict[str, set[uuid.UUID]], hidden: frozenset[uuid.UUID]) -> set[str]:
    """Every id among `ids` that belongs to a hidden project, resolved in one query per kind."""
    stmts = {
        "session_id": lambda s: select(DbSession.session_id, DbSession.project_id).where(DbSession.session_id.in_(s)),
        "frame_id": lambda s: select(Frame.frame_id, DbSession.project_id)
        .join(DbSession, DbSession.session_id == Frame.session_id).where(Frame.frame_id.in_(s)),
        "object_id": lambda s: select(Object.object_id, DbSession.project_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .join(DbSession, DbSession.session_id == Frame.session_id).where(Object.object_id.in_(s)),
        "track_id": lambda s: select(Track.track_id, DbSession.project_id)
        .join(DbSession, DbSession.session_id == Track.session_id).where(Track.track_id.in_(s)),
    }
    blocked: set[str] = set()
    async with get_sessionmaker()() as db:
        for kind, wanted in ids.items():
            if wanted:
                rows = (await db.execute(stmts[kind](list(wanted)))).all()
                blocked |= {str(i) for i, pid in rows if pid in hidden}
    return blocked


def _names_blocked(row: dict, blocked: set[str]) -> bool:
    return any(isinstance(row.get(k), str) and row[k] in blocked for k in ID_KEYS)


def _prune(node: Any, blocked: set[str]) -> Any:
    if isinstance(node, list):
        return [_prune(v, blocked) for v in node if not (isinstance(v, dict) and _names_blocked(v, blocked))]
    if isinstance(node, dict):
        return {k: _prune(v, blocked) for k, v in node.items()}
    return node


async def filter_hidden_rows(request: Request, call_next):
    response = await call_next(request)
    hidden = as_hidden(getattr(request.state, "hidden_projects", None))
    if not hidden or response.status_code != 200 \
            or not response.headers.get("content-type", "").startswith("application/json"):
        return response
    length = response.headers.get("content-length")
    if length and int(length) > MAX_BODY:
        return response

    body = b"".join([chunk async for chunk in response.body_iterator])
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    try:
        data = json.loads(body)
    except ValueError:
        return Response(body, status_code=response.status_code, headers=headers)
    ids: dict[str, set[uuid.UUID]] = {k: set() for k in ID_KEYS}
    _collect(data, ids)
    blocked = await _blocked(ids, hidden) if any(ids.values()) else set()
    if blocked:
        body = json.dumps(_prune(data, blocked), separators=(",", ":")).encode()
    return Response(body, status_code=response.status_code, headers=headers)


__all__ = ["filter_hidden_rows"]
