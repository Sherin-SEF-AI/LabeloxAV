"""The image panel's nearest-frame lookup, and the window that lets it stop asking.

Playback ticks the clock every few milliseconds. The panel used to fetch on every tick and the server
answered with no hint of how long the answer held, so a playing session drove about 100 requests a second
per panel into the rate limiter and got 429s back (2026-11-26). The answer now carries the playhead window
over which it stays the nearest frame. These tests pin that the window is exactly the set of timestamps that
resolve to the same frame, because one that is too wide shows a stale frame and one that is too narrow
brings the request storm back.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.inspector.events import nearest_frame

pytestmark = pytest.mark.db


async def _seed(db, ts_list: list[int]) -> uuid.UUID:
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="TEST-NEAR", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test"))
    await db.flush()
    for ts in ts_list:
        db.add(Frame(frame_id=uuid.uuid4(), session_id=sid, ts_ns=ts, cam_id="CAM_FRONT",
                     img_uri=f"s3://labeloxav/frames/{sid}/{ts}.jpg", width=1280, height=720))
    await db.commit()
    return sid


async def test_the_nearest_frame_and_its_window():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 2_000, 4_000])
        got = await nearest_frame(db, sid, 2_100)
    assert got["ts_ns"] == 2_000
    assert got["valid_from_ns"] == 1_501 and got["valid_to_ns"] == 3_001


async def test_every_timestamp_inside_the_window_resolves_to_the_same_frame():
    """The property the panel relies on when it skips a fetch."""
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 2_000, 4_000, 4_300])
        first = await nearest_frame(db, sid, 3_200)
        lo, hi = first["valid_from_ns"], first["valid_to_ns"]
        for ts in (lo, (lo + hi) // 2, hi - 1):
            assert (await nearest_frame(db, sid, ts))["frame_id"] == first["frame_id"]
        # and the first timestamp past it is somebody else's frame
        assert (await nearest_frame(db, sid, hi))["frame_id"] != first["frame_id"]


async def test_the_window_is_open_at_the_ends_of_the_session():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 2_000])
        before = await nearest_frame(db, sid, 0)
        after = await nearest_frame(db, sid, 99_999)
    assert before["ts_ns"] == 1_000 and before["valid_from_ns"] is None and before["valid_to_ns"] == 1_501
    assert after["ts_ns"] == 2_000 and after["valid_to_ns"] is None and after["valid_from_ns"] == 1_501


async def test_the_frame_ahead_wins_when_it_is_closer():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 2_000, 2_400])
        got = await nearest_frame(db, sid, 1_900)
    assert got["ts_ns"] == 2_000 and got["valid_from_ns"] == 1_501 and got["valid_to_ns"] == 2_201


async def test_a_playhead_exactly_halfway_goes_to_the_earlier_frame_and_the_window_agrees():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 2_000])
        mid = await nearest_frame(db, sid, 1_500)
    assert mid["ts_ns"] == 1_000 and mid["valid_to_ns"] == 1_501


async def test_a_session_with_no_frames_returns_nothing():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [])
        assert await nearest_frame(db, sid, 5) is None
