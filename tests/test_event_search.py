"""Asking the corpus a question about behaviour.

Every other event route is scoped to a session, which is right for reviewing a drive and wrong for the
question the events exist to answer. The case that matters here is the conjunction, "a crossing *while* a
signal was red", because a filtered list is something the session routes could already almost do and a
temporal join is not.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Frame, TimelineEvent
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.intelligence.event_search import corpus_summary, search_events

MS = 1_000_000


async def _session(city="BLR", vehicle="veh-search"):
    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id=vehicle, city=city, start_ts_ns=0, end_ts_ns=10**10,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.commit()
        return s.session_id, f.frame_id


async def _event(sid, fid, kind, t0, t1=None, payload=None, conf=0.7, state="review"):
    async with get_sessionmaker()() as db:
        e = TimelineEvent(session_id=sid, frame_id=fid, kind=kind, modality="driving",
                          t_start_ns=t0, t_end_ns=t1, payload=payload or {}, conf=conf,
                          source="auto", state=state)
        db.add(e)
        await db.commit()
        return e.event_id


@pytest.mark.db
async def test_a_crossing_while_the_signal_was_red_is_findable_and_one_during_green_is_not():
    """The query the layer exists for, and the one no session-scoped route could answer."""
    sid, fid = await _session()
    # A crossing overlapping a red phase, and another overlapping a green one.
    await _event(sid, fid, "lane_change_illegal", 1000 * MS, 1400 * MS)
    await _event(sid, fid, "signal_phase", 900 * MS, 1500 * MS, {"state": "R"})
    await _event(sid, fid, "lane_change_illegal", 5000 * MS, 5400 * MS)
    await _event(sid, fid, "signal_phase", 4900 * MS, 5500 * MS, {"state": "G"})

    async with get_sessionmaker()() as db:
        both = await search_events(db, kinds=["lane_change_illegal"], session_id=sid)
        on_red = await search_events(db, kinds=["lane_change_illegal"], session_id=sid,
                                     with_kind="signal_phase", with_payload_state=["R"])
        on_green = await search_events(db, kinds=["lane_change_illegal"], session_id=sid,
                                       with_kind="signal_phase", with_payload_state=["G"])
    assert both["total"] == 2
    assert on_red["total"] == 1
    assert on_green["total"] == 1
    assert on_red["results"][0]["t_start_ns"] == 1000 * MS


@pytest.mark.db
async def test_a_crossing_with_no_overlapping_phase_is_excluded():
    """Overlap, not merely co-presence in the same session. Without this the conjunction would return
    every crossing in any drive that happened to pass a signal."""
    sid, fid = await _session()
    await _event(sid, fid, "lane_change", 1000 * MS, 1100 * MS)
    await _event(sid, fid, "signal_phase", 8000 * MS, 8500 * MS, {"state": "R"})
    async with get_sessionmaker()() as db:
        got = await search_events(db, kinds=["lane_change"], session_id=sid,
                                  with_kind="signal_phase", with_payload_state=["R"])
    assert got["total"] == 0


@pytest.mark.db
async def test_the_window_lets_while_mean_around():
    """A crossing a second after the light changed is still a crossing at that light."""
    sid, fid = await _session()
    await _event(sid, fid, "lane_change", 2000 * MS, 2100 * MS)
    await _event(sid, fid, "signal_phase", 1000 * MS, 1500 * MS, {"state": "R"})
    async with get_sessionmaker()() as db:
        tight = await search_events(db, kinds=["lane_change"], session_id=sid,
                                    with_kind="signal_phase", with_payload_state=["R"])
        wide = await search_events(db, kinds=["lane_change"], session_id=sid,
                                   with_kind="signal_phase", with_payload_state=["R"],
                                   within_ns=1000 * MS)
    assert tight["total"] == 0
    assert wide["total"] == 1


@pytest.mark.db
async def test_severity_is_expanded_to_kinds_so_the_filter_survives_the_limit():
    """Severity is a property of the kind in the taxonomy, not a column. Filtering it after the fetch would
    return whatever fraction of the page happened to match."""
    sid, fid = await _session()
    for _ in range(6):
        await _event(sid, fid, "signal_phase", 10 * MS, 20 * MS, {"state": "G"})
    await _event(sid, fid, "lane_change_illegal", 30 * MS, 40 * MS)
    async with get_sessionmaker()() as db:
        got = await search_events(db, session_id=sid, severities=["violation"], limit=2)
    assert got["total"] == 1
    assert got["results"][0]["kind"] == "lane_change_illegal"


@pytest.mark.db
async def test_a_result_carries_the_context_needed_to_go_and_look_at_it():
    """An event with a session, a frame and an offset is a place to go. One that is only a row is a
    statistic."""
    sid, fid = await _session(city="PUNE", vehicle="TIGOR-99")
    await _event(sid, fid, "lane_change_illegal", 12_300 * MS, 12_900 * MS)
    async with get_sessionmaker()() as db:
        got = await search_events(db, session_id=sid, kinds=["lane_change_illegal"])
    r = got["results"][0]
    assert r["city"] == "PUNE" and r["vehicle_id"] == "TIGOR-99"
    assert r["frame_id"] == str(fid)
    assert r["severity"] == "violation"
    # Frames start at ts 0 in this fixture, so the offset is the raw time.
    assert r["at_s"] == pytest.approx(12.3, abs=0.01)
    assert r["duration_s"] == pytest.approx(0.6, abs=0.01)


@pytest.mark.db
async def test_an_unknown_severity_returns_nothing_and_says_why():
    async with get_sessionmaker()() as db:
        got = await search_events(db, severities=["catastrophic"])
    assert got["total"] == 0
    assert "no event kind carries severity" in got["detail"]


@pytest.mark.db
async def test_the_corpus_summary_counts_by_kind_severity_and_city():
    sid, fid = await _session(city="CHENNAI")
    await _event(sid, fid, "lane_change_illegal", 1 * MS, 2 * MS)
    async with get_sessionmaker()() as db:
        out = await corpus_summary(db)
    assert out["total"] >= 1
    assert out["by_severity"].get("violation", 0) >= 1
    assert "CHENNAI" in out["cities"]
    assert out["sessions_with_events"] >= 1
    assert uuid.UUID(str(sid))
