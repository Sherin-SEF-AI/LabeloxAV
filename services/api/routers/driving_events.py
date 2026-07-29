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


@router.post("/sessions/{session_id}/lanes/classify", dependencies=[Depends(require_role("annotator"))])
async def classify_lanes(session_id: UUID, apply: bool = True, reclassify: bool = False,
                         db: AsyncSession = Depends(db_session)):
    """Read each lane's type off the frame it was drawn on.

    The other half of the lane prerequisite, alongside linking. Linking says which crossings exist; typing
    says which of them are offences. A lane a person typed is never overwritten, and `reclassify` decides
    whether lanes already measured are looked at again, which is what a threshold change needs.
    """
    from services.intelligence.lane_typing import classify_session_lanes

    return await classify_session_lanes(db, session_id, apply=apply, reclassify=reclassify)


@router.get("/lanes/type-coverage")
async def lane_type_coverage(db: AsyncSession = Depends(db_session)):
    """How much of the corpus carries a measured lane type rather than the old hardcoded default."""
    from services.intelligence.lane_typing import corpus_type_summary

    return await corpus_type_summary(db)


@router.get("/events/search")
async def search(kind: str | None = None, severity: str | None = None, state: str | None = None,
                 city: str | None = None, session_id: UUID | None = None,
                 min_conf: float | None = None,
                 with_kind: str | None = None, with_state: str | None = None,
                 within_ms: int = 0,
                 limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0),
                 db: AsyncSession = Depends(db_session)):
    """Behaviour across the whole corpus, not one session.

    Every other event route is session-scoped, which is right for reviewing a drive and wrong for the
    question the events exist to answer. `with_kind` and `with_state` are the conjunction: "every illegal
    lane change while a signal was showing red" is a temporal join within a session, and it is the query
    that makes this layer worth having rather than merely correct.

    Comma-separated values are accepted on kind, severity and state so a filter chip can send several.
    """
    from services.intelligence.event_search import search_events

    def many(v):
        return [x.strip() for x in v.split(",") if x.strip()] if v else None

    return await search_events(
        db, kinds=many(kind), severities=many(severity), states=many(state), city=city,
        session_id=session_id, min_conf=min_conf, with_kind=with_kind,
        with_payload_state=many(with_state), within_ns=int(within_ms) * 1_000_000,
        limit=limit, offset=offset)


@router.get("/events/corpus-summary")
async def corpus_summary_ep(db: AsyncSession = Depends(db_session)):
    """What the corpus holds, so a question can be aimed rather than guessed at."""
    from services.intelligence.event_search import corpus_summary

    return await corpus_summary(db)


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
