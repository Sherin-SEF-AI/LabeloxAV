"""Track events: typed spans within a track.

The extent is what `Track.intents` never had. An intent says a track cut in; an event says it cut in over
frames 40 to 55 of 93, which is what lets a reviewer find the moment, an export cut the clip, and the
coverage datasheet count how much of the corpus contains one.

Two things are checked here that the schema cannot express:

  - the span's frames belong to the same session as the track. A foreign key can say `frame` and `track`
    exist; it cannot say they are the same drive, and an event spanning two sessions is not wrong-looking in
    the database, it is silently unqueryable.
  - the event type belongs to the pack the session was captured under, and applies to that track's
    superclass. Freezing the AV vocabulary into a check constraint would make a second domain's events
    invalid rows, so the pack is the authority and this is where it is consulted.

The floor is reviewer, declared on the router. services/api/routers/tracks.py has no floor at all, which is
how `relabel_track` became a review path an annotator could use to accept a whole track; a new surface does
not repeat that.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Track, TrackEvent
from services.api.deps import current_user, db_session, require_role
from services.autolabel.ontology import get_ontology
from services.domain import pack_id_for_session, track_event_spec, validate_track_event_type

log = get_logger("api_track_events")
router = APIRouter(dependencies=[Depends(require_role("reviewer"))])

_STATES = {"proposed", "accepted", "rejected"}


class TrackEventIn(BaseModel):
    event_type: str
    start_frame_id: UUID
    end_frame_id: UUID
    notes: str | None = None
    # A person drawing a span is asserting it, so it lands accepted. A proposer uses the module in
    # services/autolabel/event_proposals.py and writes `proposed` directly, never through this route.
    state: str = "accepted"


class TrackEventPatch(BaseModel):
    state: str | None = None
    notes: str | None = None
    start_frame_id: UUID | None = None
    end_frame_id: UUID | None = None


class TrackEventOut(BaseModel):
    event_id: str
    track_id: str
    event_type: str
    start_frame_id: str
    end_frame_id: str
    start_ts_ns: int
    end_ts_ns: int
    source: str
    state: str
    confidence: float | None = None
    evidence: dict = Field(default_factory=dict)
    notes: str | None = None
    created_by: str | None = None


def _out(e: TrackEvent) -> dict:
    return TrackEventOut(
        event_id=str(e.event_id), track_id=str(e.track_id), event_type=e.event_type,
        start_frame_id=str(e.start_frame_id), end_frame_id=str(e.end_frame_id),
        start_ts_ns=e.start_ts_ns, end_ts_ns=e.end_ts_ns, source=e.source, state=e.state,
        confidence=e.confidence, evidence=e.evidence or {}, notes=e.notes, created_by=e.created_by,
    ).model_dump()


async def _resolve_span(db: AsyncSession, track: Track, start_id: UUID, end_id: UUID) -> tuple[Frame, Frame]:
    """The two frames of a span, checked to be this track's session and ordered in time.

    Ordered rather than refused when reversed: a drag runs in whichever direction the annotator moved, and
    refusing a right-to-left drag is a bug report from every annotator who makes one.
    """
    frames = {
        f.frame_id: f for f in
        (await db.execute(select(Frame).where(Frame.frame_id.in_({start_id, end_id})))).scalars()
    }
    missing = sorted(str(i) for i in {start_id, end_id} if i not in frames)
    if missing:
        raise HTTPException(404, f"frame not found: {', '.join(missing)}")
    a, b = frames[start_id], frames[end_id]
    wrong = [str(f.frame_id) for f in (a, b) if f.session_id != track.session_id]
    if wrong:
        raise HTTPException(400, {"detail": "span frames must belong to the track's session",
                                  "track_session_id": str(track.session_id), "foreign_frames": wrong})
    return (a, b) if a.ts_ns <= b.ts_ns else (b, a)


@router.get("/tracks/{track_id}/changepoints")
async def track_changepoints_ep(track_id: str, source: str = "object_speed",
                                db: AsyncSession = Depends(db_session)):
    """Where this track's motion actually changes, as targets a span edge can snap to.

    There was no changepoint detection anywhere in this repo. The nearest existing things answer a
    different question: a threshold crossing is where a behaviour got big enough to notice, not where it
    started.

    `source=ego_speed` is the natural signal and refuses on this corpus, with the count: ego speed is set
    on 6 frames of 41,752. It refuses rather than falling back, because a snap that silently used a
    different signal would put span edges where nobody could explain them.

    `source=object_speed` works, on 252,815 samples. Its noise is comparable to the events it is asked to
    find, so every shift has to clear the series' own scatter and anything a slope explains better is
    refused. Over 400 real tracks that leaves 86% with no changepoint at all.
    """
    import uuid as _uuid

    from services.temporal.changepoint import track_changepoints

    try:
        return await track_changepoints(db, _uuid.UUID(track_id), source=source)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/tracks/{track_id}/events")
async def list_events(track_id: UUID, db: AsyncSession = Depends(db_session)):
    """Every event on a track, earliest first, with the pack's vocabulary alongside.

    The vocabulary rides with the list so the picker does not need a second request, and so the definitions
    an annotator reads always come from the pack this track's session was captured under.
    """
    track = await db.get(Track, track_id)
    if track is None:
        raise HTTPException(404, "track not found")
    rows = (await db.execute(
        select(TrackEvent).where(TrackEvent.track_id == track_id)
        .order_by(TrackEvent.start_ts_ns, TrackEvent.created_at))).scalars().all()
    pack_id = await pack_id_for_session(db, track.session_id)
    spec = track_event_spec(pack_id)
    onto = get_ontology()
    try:
        l1 = onto.by_id(track.class_id).l1
    except KeyError:
        l1 = None
    types = [] if spec is None else [
        {"name": t.name, "definition": t.definition, "applies_to": t.applies_to,
         "proposable": t.proposable,
         # Computed here rather than in the client so one rule decides what is offered and what is accepted.
         "applicable": not validate_track_event_type(t.name, l1, pack_id)}
        for t in spec.types
    ]
    return {"track_id": str(track_id), "class_l1": l1, "events": [_out(e) for e in rows],
            "event_types": types}


@router.post("/tracks/{track_id}/events", status_code=201)
async def create_event(track_id: UUID, payload: TrackEventIn,
                       db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    track = await db.get(Track, track_id)
    if track is None:
        raise HTTPException(404, "track not found")
    if payload.state not in _STATES:
        raise HTTPException(400, f"state must be one of {sorted(_STATES)}")

    pack_id = await pack_id_for_session(db, track.session_id)
    onto = get_ontology()
    try:
        l1 = onto.by_id(track.class_id).l1
    except KeyError:
        l1 = None
    errors = validate_track_event_type(payload.event_type, l1, pack_id)
    if errors:
        raise HTTPException(400, {"event_errors": errors})

    start, end = await _resolve_span(db, track, payload.start_frame_id, payload.end_frame_id)
    ev = TrackEvent(
        track_id=track_id, event_type=payload.event_type,
        start_frame_id=start.frame_id, end_frame_id=end.frame_id,
        start_ts_ns=start.ts_ns, end_ts_ns=end.ts_ns,
        source="human", state=payload.state, notes=payload.notes,
        created_by=getattr(user, "name", None),
        # Stamped so an event drawn under one class vocabulary is still readable after a bump, the same way
        # a gold set records the ontology it was sealed under.
        ontology_version=onto.version,
    )
    db.add(ev)
    await db.commit()
    log.info("track_event.created", track_id=str(track_id), event_type=payload.event_type)
    return _out(ev)


@router.patch("/track-events/{event_id}")
async def update_event(event_id: UUID, payload: TrackEventPatch,
                       db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Accept, reject, retime or annotate one event.

    Accepting a proposal keeps `source=heuristic`: the record of who suggested it is what makes the
    proposer's precision measurable later, and overwriting it with `human` would make every accepted
    proposal indistinguishable from a hand-drawn span.
    """
    ev = await db.get(TrackEvent, event_id)
    if ev is None:
        raise HTTPException(404, "event not found")
    if payload.state is not None:
        if payload.state not in _STATES:
            raise HTTPException(400, f"state must be one of {sorted(_STATES)}")
        ev.state = payload.state
    if payload.notes is not None:
        ev.notes = payload.notes
    if payload.start_frame_id is not None or payload.end_frame_id is not None:
        track = await db.get(Track, ev.track_id)
        start, end = await _resolve_span(
            db, track, payload.start_frame_id or ev.start_frame_id, payload.end_frame_id or ev.end_frame_id)
        ev.start_frame_id, ev.end_frame_id = start.frame_id, end.frame_id
        ev.start_ts_ns, ev.end_ts_ns = start.ts_ns, end.ts_ns
    await db.commit()
    return _out(ev)


@router.delete("/track-events/{event_id}", dependencies=[Depends(require_role("admin"))])
async def delete_event(event_id: UUID, db: AsyncSession = Depends(db_session)):
    """Admin only. Rejecting an event is the reviewer's action and it keeps the record of the disagreement;
    deleting removes the evidence that anybody ever thought the span was there."""
    ev = await db.get(TrackEvent, event_id)
    if ev is None:
        raise HTTPException(404, "event not found")
    await db.delete(ev)
    await db.commit()
    return {"deleted": str(event_id)}
