"""Find a thing by its name or its id: the data half of the command palette.

The palette found pages and nothing else, and /search is visual similarity. A session could be found only
by scrolling the sessions page, and an id copied from a log line, a review queue or a teammate's message led
nowhere. This answers "where is DEL-01" and "open 5901b4fa" from the same box.

Sessions match on vehicle, city, route or id. Frames, objects and tracks match by id prefix only, and only
when the query looks like an id, so a word typed into the palette never scans six hundred thousand objects.
Every result carries the page it opens.
"""

from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object, OntologyClass, Track
from db.models import Session as DbSession
from services.api.deps import db_session, require_role
from services.api.project_guard import caller_hidden_projects
from services.project_access import as_hidden, drop_hidden, project_of, scope_visible_sessions

router = APIRouter()

PER_KIND = 6
# An id prefix this short already narrows a 600k-row table to a handful; anything shorter is a word.
MIN_ID_PREFIX = 6
_ID_LIKE = re.compile(r"^[0-9a-f-]+$")


def _id_prefix(q: str) -> str | None:
    s = q.strip().lower()
    return s if len(s.replace("-", "")) >= MIN_ID_PREFIX and _ID_LIKE.match(s) else None


def _uuid_range(prefix: str) -> tuple[uuid.UUID, uuid.UUID]:
    """The ids an id prefix covers, as a range the primary-key index answers directly. Comparing ids as text
    instead scans every row: 1.4 s over this corpus's 606k objects, against milliseconds this way."""
    hexes = prefix.replace("-", "")[:32]
    return uuid.UUID(hexes.ljust(32, "0")), uuid.UUID(hexes.ljust(32, "f"))


async def _first_frame(db: AsyncSession, session_id) -> str | None:
    fid = (await db.execute(select(Frame.frame_id).where(Frame.session_id == session_id)
                            .order_by(Frame.ts_ns).limit(1))).scalar_one_or_none()
    return str(fid) if fid else None


@router.get("/find", dependencies=[Depends(require_role("annotator"))])
async def find(q: str = Query(min_length=2, max_length=80), db: AsyncSession = Depends(db_session),
               hidden=Depends(caller_hidden_projects)):
    hidden = as_hidden(hidden)
    text = q.strip()
    like = f"%{text.replace('%', '').replace('_', '')}%"
    prefix = _id_prefix(text)
    out: list[dict] = []

    session_match = [DbSession.vehicle_id.ilike(like), DbSession.city.ilike(like), DbSession.route.ilike(like)]
    if prefix:
        lo, hi = _uuid_range(prefix)
        session_match.append(DbSession.session_id.between(lo, hi))
    sessions = (await db.execute(scope_visible_sessions(select(DbSession), hidden).where(or_(*session_match))
                                 .order_by(DbSession.start_ts_ns.desc()).limit(PER_KIND))).scalars().all()
    for s in sessions:
        first = await _first_frame(db, s.session_id)
        out.append({"kind": "session", "id": str(s.session_id),
                    "label": " · ".join(p for p in (s.vehicle_id, s.city) if p) or str(s.session_id)[:8],
                    "detail": f"{str(s.session_id)[:8]}{' · ' + s.route if s.route else ''}",
                    "href": f"/frame/{first}" if first else "/annotations"})

    if prefix:
        lo, hi = _uuid_range(prefix)
        frames = (await db.execute(select(Frame.frame_id, Frame.cam_id, DbSession.vehicle_id)
                                   .join(DbSession, DbSession.session_id == Frame.session_id)
                                   .where(Frame.frame_id.between(lo, hi)).limit(PER_KIND))).all()
        for fid, cam, vehicle in frames:
            out.append({"kind": "frame", "id": str(fid), "label": f"Frame {str(fid)[:8]}",
                        "detail": f"{vehicle} · {cam}", "href": f"/frame/{fid}"})

        objects = (await db.execute(select(Object.object_id, Object.frame_id, OntologyClass.name, Object.state)
                                    .join(OntologyClass, OntologyClass.id == Object.class_id)
                                    .where(Object.object_id.between(lo, hi)).limit(PER_KIND))).all()
        for oid, fid, cls, state in objects:
            out.append({"kind": "object", "id": str(oid), "label": f"{cls} {str(oid)[:8]}",
                        "detail": f"{state} · frame {str(fid)[:8]}", "href": f"/frame/{fid}?focus={oid}"})

        tracks = (await db.execute(select(Track.track_id, OntologyClass.name, func.count())
                                   .join(OntologyClass, OntologyClass.id == Track.class_id)
                                   .outerjoin(Object, Object.track_id == Track.track_id)
                                   .where(Track.track_id.between(lo, hi))
                                   .group_by(Track.track_id, OntologyClass.name).limit(PER_KIND))).all()
        for tid, cls, n in tracks:
            out.append({"kind": "track", "id": str(tid), "label": f"{cls} track {str(tid)[:8]}",
                        "detail": f"{n} objects", "href": f"/track/{tid}"})

    if hidden:
        # Sessions were scoped in their query. Frames, objects and tracks found by id prefix are checked here, and
        # any in a project hidden from the caller dropped, keeping the order results were found in.
        keep: set[tuple[str, str]] = set()
        for kind, key in (("frame", "frame_id"), ("object", "object_id")):
            rows = [{"id": r["id"], key: r["id"]} for r in out if r["kind"] == kind]
            keep |= {(kind, r["id"]) for r in await drop_hidden(db, hidden, rows, key)}
        for r in out:
            if r["kind"] == "track":
                found, pid = await project_of(db, "track_id", r["id"])
                if not (found and pid in hidden):
                    keep.add(("track", r["id"]))
        out = [r for r in out if r["kind"] == "session" or (r["kind"], r["id"]) in keep]
    return {"q": text, "results": out}
