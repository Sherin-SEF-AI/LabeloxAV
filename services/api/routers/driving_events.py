"""Driving events: the vocabulary, deriving them, and ruling on them.

The event CRUD in `inertial.py` stays where it is and keeps serving the imu, audio and scene modalities it
was written for. What lives here is everything specific to driving behaviour: the taxonomy the editor renders
from, the derivers, the filtered read the review surfaces use, and confirm/reject, which is the whole point.
A derived event is a candidate, and a candidate nobody ruled on is not a label.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()


@router.get("/events/taxonomy")
async def taxonomy():
    """The event vocabulary: kinds, their shape, anchor, severity, and whether a deriver may propose them.

    The editor builds its event picker from this, so adding a kind to the config file adds it to the UI with
    no code change on either side.
    """
    from services.intelligence.event_taxonomy import describe

    return describe()


@router.get("/sessions/{session_id}/driving-events")
async def list_driving_events(session_id: UUID, kind: str | None = None, state: str | None = None,
                              severity: str | None = None, track_id: UUID | None = None,
                              limit: int = Query(2000, ge=1, le=20000)):
    """One session's driving events, narrowed by kind, review state, severity, or actor."""
    from services.intelligence.timeline_events import list_events

    return await list_events(session_id, modality="driving", kind=kind, state=state,
                             track_id=track_id, severity=severity, limit=limit)


@router.get("/sessions/{session_id}/driving-events/summary")
async def summary(session_id: UUID, db: AsyncSession = Depends(db_session)):
    """Counts by kind, severity and review state. What a session header shows without pulling every event."""
    from services.intelligence.driving_events import session_event_summary

    return await session_event_summary(db, session_id)


@router.post("/sessions/{session_id}/driving-events/derive",
             dependencies=[Depends(require_role("annotator"))])
async def derive(session_id: UUID, prune_stale: bool = True,
                 db: AsyncSession = Depends(db_session)):
    """Derive lane and signal events for a session and reconcile them with what is stored.

    Safe to re-run: matching candidates update in place rather than duplicating, human decisions are never
    touched, and candidates the geometry no longer implies are pruned unless prune_stale is turned off.
    """
    from services.intelligence.driving_events import persist_driving_events

    return await persist_driving_events(db, session_id, prune_stale=prune_stale)


@router.post("/sessions/{session_id}/driving-events/preview",
             dependencies=[Depends(require_role("annotator"))])
async def preview(session_id: UUID, db: AsyncSession = Depends(db_session)):
    """What deriving would produce, without writing anything.

    The way to try a changed threshold in the taxonomy against a real session before letting it rewrite the
    session's candidates, which is the same posture the reasoner's rerun takes for the same reason.
    """
    from services.intelligence.driving_events import derive_events

    events = await derive_events(db, session_id)
    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return {"session_id": str(session_id), "derived": len(events),
            "by_kind": dict(sorted(counts.items())), "events": events[:200],
            "truncated": max(0, len(events) - 200)}


@router.post("/sessions/{session_id}/lanes/link", dependencies=[Depends(require_role("annotator"))])
async def link_lanes(session_id: UUID, apply: bool = True, db: AsyncSession = Depends(db_session)):
    """Give a session's lanes an identity across frames, from the control points alone.

    A prerequisite for any lane behaviour rather than a feature in itself, which is why the derive route runs
    it on demand. Exposed separately so the result can be inspected before deriving, and because a session
    whose lanes were corrected by hand is worth relinking without waiting for the next derivation.
    """
    from services.intelligence.lane_linking import link_lanes_for_session

    return await link_lanes_for_session(db, session_id, apply=apply)


class RulingIn(BaseModel):
    """A human decision on a candidate. note is free text kept on the event for the next reader."""

    note: str | None = None
    expected_version: int | None = None


async def _rule(event_id: UUID, state: str, body: RulingIn, db: AsyncSession) -> dict:
    from db.models import TimelineEvent
    from services.intelligence.timeline_events import event_row

    e = await db.get(TimelineEvent, event_id)
    if e is None:
        raise HTTPException(404, "event not found")
    if body.expected_version is not None and e.version != body.expected_version:
        raise HTTPException(409, {"detail": "event changed since you loaded it",
                                  "current_version": e.version})
    e.state = state
    if body.note:
        # Kept on provenance rather than payload: payload is the deriver's output and a re-derivation
        # overwrites it, which would silently discard the reviewer's reason for the ruling.
        e.provenance = {**(e.provenance or {}), "review_note": body.note}
    e.version += 1
    await db.commit()
    await db.refresh(e)
    return event_row(e)


@router.post("/driving-events/{event_id}/confirm", dependencies=[Depends(require_role("reviewer"))])
async def confirm(event_id: UUID, body: RulingIn, db: AsyncSession = Depends(db_session)):
    """Accept a candidate. It stops being re-derivable and becomes a label."""
    return await _rule(event_id, "confirmed", body, db)


@router.post("/driving-events/{event_id}/reject", dependencies=[Depends(require_role("reviewer"))])
async def reject(event_id: UUID, body: RulingIn, db: AsyncSession = Depends(db_session)):
    """Reject a candidate. It is kept, not deleted, so a re-derivation does not propose it again."""
    return await _rule(event_id, "rejected", body, db)


class DrivingEventIn(BaseModel):
    """A human-authored driving event. Validated against the taxonomy before it is written."""

    kind: str
    t_start_ns: int
    t_end_ns: int | None = None
    track_id: UUID | None = None
    frame_id: UUID | None = None
    payload: dict = {}


@router.post("/sessions/{session_id}/driving-events",
             dependencies=[Depends(require_role("annotator"))])
async def create(session_id: UUID, body: DrivingEventIn):
    """Author a driving event by hand, including the human-only kinds no deriver may propose."""
    from services.intelligence.event_taxonomy import TaxonomyError
    from services.intelligence.timeline_events import create_event

    try:
        return await create_event(session_id, body.kind, "driving", body.t_start_ns, body.t_end_ns,
                                  body.payload, source="human", track_id=body.track_id,
                                  frame_id=body.frame_id)
    except TaxonomyError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/tracks/{track_id}/driving-events")
async def track_events(track_id: UUID, db: AsyncSession = Depends(db_session)):
    """Everything derived about one actor, in time order.

    What the track inspector shows: the behaviour of this vehicle over the clip, rather than a list of the
    boxes that make it up.
    """
    from sqlalchemy import select

    from db.models import TimelineEvent
    from services.intelligence.timeline_events import event_row

    rows = (await db.execute(
        select(TimelineEvent).where(TimelineEvent.track_id == track_id)
        .order_by(TimelineEvent.t_start_ns))).scalars().all()
    return {"track_id": str(track_id), "count": len(rows), "events": [event_row(e) for e in rows]}
