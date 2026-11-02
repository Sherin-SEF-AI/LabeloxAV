"""Tube scores on stored tracks: a steady track scores high, a jumpy one lower, a short one is unmeasured."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed(boxes_by_track: dict[str, list[list[float] | None]], n_frames: int):
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name("sedan").id
    sid, ts = uuid.uuid4(), now_ns()
    tids = {}
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TUBE", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(n_frames),
                         city="BLR", sensors={}, ontology_version=onto.version))
        await db.flush()
        fids = []
        for i in range(n_frames):
            fid = uuid.uuid4()
            fids.append(fid)
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                         img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        for name, boxes in boxes_by_track.items():
            tid = uuid.uuid4()
            tids[name] = tid
            db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts, last_ts_ns=ts + seconds_to_ns(n_frames)))
            await db.flush()
            for fid, b in zip(fids, boxes, strict=False):
                if b is not None:
                    db.add(Object(object_id=uuid.uuid4(), frame_id=fid, track_id=tid, class_id=cid, bbox=b,
                                  conf=0.8, source="fused", state="review", attrs={}, provenance={}, version=1))
        await db.commit()
    return sid, tids


def test_steady_tracks_score_higher_than_jumpy_ones_and_short_ones_are_unmeasured():
    from sqlalchemy import select

    from db.models import Track
    from db.session import get_sessionmaker
    from services.temporal.tube import score_session_tracks

    n = 10
    steady = [[100.0 + 3 * i, 100.0, 200.0 + 3 * i, 180.0] for i in range(n)]
    jumpy = [[100.0 + (90 if i % 2 else 0), 100.0, 200.0 + (40 if i % 3 else 0), 180.0 + (60 if i % 2 else 0)]
             if i % 4 != 2 else None for i in range(n)]
    short = [[500.0, 500.0, 560.0, 540.0], [503.0, 500.0, 563.0, 540.0]] + [None] * (n - 2)
    sid, tids = _run(_seed({"steady": steady, "jumpy": jumpy, "short": short}, n))

    async def _go():
        async with get_sessionmaker()() as db:
            res = await score_session_tracks(db, sid)
            await db.commit()
            scores = {tid: s for tid, s in (await db.execute(
                select(Track.track_id, Track.tube_score).where(Track.session_id == sid))).all()}
        return res, scores

    res, scores = _run(_go())
    assert res == {"tracks": 3, "measured": 2}
    assert scores[tids["short"]] is None
    assert scores[tids["steady"]] > scores[tids["jumpy"]]
