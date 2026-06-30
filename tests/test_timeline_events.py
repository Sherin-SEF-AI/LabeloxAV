"""Milestone B: timeline events. Pure overlap logic, optimistic-concurrency CRUD, and the invariant that
auto-detected inertial events persist as unconfirmed candidates (never auto-accepted)."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select

from db.models import Session as DbSession
from db.models import TimelineEvent
from db.session import get_sessionmaker
from services.intelligence.timeline_events import (
    create_event,
    delete_event,
    events_overlap,
    persist_auto_inertial_events,
    update_event,
)


def test_events_overlap_ranges_and_points():
    assert events_overlap(0, 100, 50, 150)        # overlapping ranges
    assert not events_overlap(0, 100, 200, 300)   # disjoint
    assert events_overlap(0, 100, 100, 200)       # touch at the boundary
    assert events_overlap(50, None, 0, 100)       # a point inside a range
    assert not events_overlap(150, None, 0, 100)  # a point outside


async def test_event_crud_with_optimistic_concurrency():
    async with get_sessionmaker()() as db:
        sid = (await db.execute(select(DbSession.session_id).limit(1))).scalar()
    assert sid is not None
    ev = await create_event(sid, "hard_brake", "imu", 1000, 2000, {"peak": -5.0}, source="human")
    eid = uuid.UUID(ev["event_id"])
    try:
        assert ev["version"] == 1 and ev["state"] == "confirmed"
        upd = await update_event(eid, {"state": "rejected"}, expected_version=1)
        assert upd["version"] == 2 and upd["state"] == "rejected"
        conflict = await update_event(eid, {"state": "confirmed"}, expected_version=1)   # stale
        assert conflict.get("conflict") and conflict["current_version"] == 2
    finally:
        await delete_event(eid, expected_version=None)


async def test_auto_inertial_events_persist_as_unconfirmed():
    async with get_sessionmaker()() as db:
        sid = (await db.execute(select(DbSession.session_id).limit(1))).scalar()
    res = await persist_auto_inertial_events(sid)
    assert res["all_unconfirmed"] is True
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(TimelineEvent).where(
            TimelineEvent.session_id == sid, TimelineEvent.source == "auto",
            TimelineEvent.modality == "imu"))).scalars().all()
        assert all(r.state == "review" for r in rows)   # vacuously true if the session had no spikes
        await db.execute(delete(TimelineEvent).where(
            TimelineEvent.session_id == sid, TimelineEvent.source == "auto"))
        await db.commit()
