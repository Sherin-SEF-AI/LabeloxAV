"""Milestone B: events on the canonical timeline. Create, edit, and delete human events with optimistic
concurrency (a stale write is a conflict, the same posture as Object). Persist the auto-detected inertial
spikes as unconfirmed candidates (source=auto, state=review), never accepted automatically. Bind an inertial
spike, a frame, and an audio region at one instant into a single crossmodal event, the primitive that lets a
pothole impact in IMU point the labeler at the road defect in vision.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("timeline_events")


def events_overlap(a_start: int, a_end: int | None, b_start: int, b_end: int | None) -> bool:
    """Whether two timeline events overlap in time. A point event (end is None) is the instant [start, start]."""
    a1, a2 = a_start, (a_start if a_end is None else a_end)
    b1, b2 = b_start, (b_start if b_end is None else b_end)
    return a1 <= b2 and b1 <= a2


def event_row(e) -> dict:
    from services.intelligence.event_taxonomy import severity_of

    return {"event_id": str(e.event_id), "session_id": str(e.session_id), "kind": e.kind,
            "modality": e.modality, "t_start_ns": e.t_start_ns, "t_end_ns": e.t_end_ns,
            "track_id": str(e.track_id) if e.track_id else None,
            "frame_id": str(e.frame_id) if e.frame_id else None,
            "conf": e.conf, "severity": severity_of(e.kind),
            "payload": e.payload, "source": e.source, "state": e.state, "version": e.version}


async def create_event(session_id, kind: str, modality: str, t_start_ns: int, t_end_ns: int | None,
                       payload: dict, source: str = "human", track_id=None, frame_id=None,
                       conf: float | None = None) -> dict:
    """Place an event on a session's timeline.

    Driving events are validated against the taxonomy and the others are not, deliberately. The imu, audio
    and scene kinds predate the taxonomy and are produced by detectors with their own vocabularies; forcing
    them through it now would reject events the system has been writing correctly for milestones.
    """
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.event_taxonomy import validate

    if modality == "driving":
        validate(kind, t_start_ns=t_start_ns, t_end_ns=t_end_ns, track_id=track_id,
                 frame_id=frame_id, source=source)

    async with get_sessionmaker()() as db:
        e = TimelineEvent(session_id=session_id, kind=kind, modality=modality, t_start_ns=t_start_ns,
                          t_end_ns=t_end_ns, payload=payload or {}, source=source,
                          track_id=track_id, frame_id=frame_id, conf=conf,
                          state="confirmed" if source == "human" else "review")
        db.add(e)
        await db.commit()
        await db.refresh(e)
        return event_row(e)


async def update_event(event_id, fields: dict, expected_version: int | None) -> dict:
    """Apply edits with optimistic concurrency: a stale expected_version returns a conflict (mapped to 409)."""
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        e = await db.get(TimelineEvent, event_id)
        if e is None:
            return {"error": "event not found"}
        if expected_version is not None and e.version != expected_version:
            return {"conflict": True, "current_version": e.version}
        for k in ("kind", "modality", "t_start_ns", "t_end_ns", "payload", "state"):
            if k in fields and fields[k] is not None:
                setattr(e, k, fields[k])
        e.version += 1
        await db.commit()
        await db.refresh(e)
        return event_row(e)


async def delete_event(event_id, expected_version: int | None) -> dict:
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        e = await db.get(TimelineEvent, event_id)
        if e is None:
            return {"error": "event not found"}
        if expected_version is not None and e.version != expected_version:
            return {"conflict": True, "current_version": e.version}
        await db.delete(e)
        await db.commit()
        return {"deleted": str(event_id)}


async def list_events(session_id, modality: str | None = None, kind: str | None = None,
                      state: str | None = None, track_id=None, severity: str | None = None,
                      limit: int = 2000) -> dict:
    """A session's events, narrowed.

    Severity is not a column: it is a property of the kind, held in the taxonomy so that reclassifying a kind
    does not need a data migration. It still has to be filtered in SQL rather than after the fetch, because
    filtering a limited page leaves you with whatever fraction of that page happened to match, which reads
    as "no violations in this session" when there are eleven of them. So the taxonomy is asked which kinds
    carry the severity, and that becomes an IN clause.
    """
    from sqlalchemy import select

    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.event_taxonomy import kinds as taxonomy_kinds
    from services.intelligence.event_taxonomy import severity_of

    async with get_sessionmaker()() as db:
        q = select(TimelineEvent).where(TimelineEvent.session_id == session_id)
        if modality:
            q = q.where(TimelineEvent.modality == modality)
        if kind:
            q = q.where(TimelineEvent.kind == kind)
        if state:
            q = q.where(TimelineEvent.state == state)
        if track_id:
            q = q.where(TimelineEvent.track_id == track_id)
        if severity:
            matching = [k for k in taxonomy_kinds() if severity_of(k) == severity]
            if not matching:
                return {"session_id": str(session_id), "count": 0, "events": []}
            q = q.where(TimelineEvent.kind.in_(matching))
        rows = (await db.execute(
            q.order_by(TimelineEvent.t_start_ns).limit(limit))).scalars().all()
        # Event timestamps are absolute capture time, which is what everything downstream needs and what no
        # human can read: an event at 1782458496360000000ns is 12.3 seconds into the drive, and only the
        # second form tells a reviewer where to scrub to. The session origin travels with the list so the
        # client can show the readable one without a second request per page.
        start = (await db.execute(
            select(DbSession.start_ts_ns)
            .where(DbSession.session_id == session_id))).scalar()
    return {"session_id": str(session_id), "count": len(rows),
            "session_start_ns": int(start) if start is not None else None,
            "events": [event_row(e) for e in rows]}


async def persist_auto_inertial_events(session_id) -> dict:
    """Run the derived inertial event detector and persist each spike as an unconfirmed candidate (source=
    auto, state=review). Idempotent: clears prior auto imu events for the session first."""
    from sqlalchemy import delete, select

    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.inertial_events import session_inertial_events
    res = await session_inertial_events(session_id)
    async with get_sessionmaker()() as db:
        await db.execute(delete(TimelineEvent).where(
            TimelineEvent.session_id == session_id, TimelineEvent.modality == "imu",
            TimelineEvent.source == "auto"))
        for ev in res["events"]:
            db.add(TimelineEvent(session_id=session_id, kind=ev["kind"], modality="imu",
                                 t_start_ns=ev["t_in_ns"], t_end_ns=ev["t_out_ns"],
                                 payload={"peak": ev["peak"], "severity": ev["severity"]},
                                 source="auto", state="review",
                                 provenance={"detector": "derived_inertial", "ego_source": res["source"]}))
        await db.commit()
        n = (await db.execute(select(TimelineEvent.event_id).where(
            TimelineEvent.session_id == session_id, TimelineEvent.source == "auto",
            TimelineEvent.modality == "imu"))).all()
    log.info("timeline.auto_events", session=str(session_id), candidates=len(n))
    return {"session_id": str(session_id), "candidates": len(n), "all_unconfirmed": True}


async def correlate_event(session_id, ts_ns: int, window_ns: int = 250_000_000) -> dict:
    """Bind the modalities at one instant into a crossmodal event: the nearest inertial event, the frame at
    ts, and an audio region [ts - window, ts + window]. The frame anchors the visual reference."""
    from sqlalchemy import select

    from db.models import Frame, TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.timeline import nearest_index
    async with get_sessionmaker()() as db:
        imu = (await db.execute(select(TimelineEvent).where(
            TimelineEvent.session_id == session_id, TimelineEvent.modality == "imu")
            .order_by(TimelineEvent.t_start_ns))).scalars().all()
        starts = [e.t_start_ns for e in imu]
        gi = nearest_index(starts, ts_ns)
        imu_link = event_row(imu[gi]) if gi is not None and abs(starts[gi] - ts_ns) <= window_ns else None
        fts = [int(t) for t in (await db.execute(select(Frame.ts_ns).where(
            Frame.session_id == session_id).order_by(Frame.ts_ns))).scalars().all()]
        fi = nearest_index(fts, ts_ns)
        frame_id = None
        if fi is not None:
            frame_id = str((await db.execute(select(Frame.frame_id).where(
                Frame.session_id == session_id, Frame.ts_ns == fts[fi]).limit(1))).scalar())
        payload = {"ts_ns": ts_ns, "imu_event": imu_link, "frame_id": frame_id,
                   "audio_region_ns": [ts_ns - window_ns, ts_ns + window_ns]}
        ce = TimelineEvent(session_id=session_id, kind="impact_felt_seen_heard", modality="crossmodal",
                           t_start_ns=ts_ns - window_ns, t_end_ns=ts_ns + window_ns, payload=payload,
                           source="correlated", state="review",
                           provenance={"method": "timestamp_correlation", "window_ns": window_ns})
        db.add(ce)
        await db.commit()
        await db.refresh(ce)
        return event_row(ce)
