"""The planted-error benchmark: errors are planted, detectors measured, and nothing is ever saved.

The fixture is a session the detectors can work with: three classes whose embeddings sit in separate
clusters, every label on a track of five frames. A class planted on one label is then wrong by its embedding
(confident learning) and by its track (consistency), so both should catch most plants.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
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


def _seed(n_tracks_per_class: int = 8, frames_per_track: int = 5) -> str:
    from db.models import Frame, Object, ObjectEmbedding, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    four = [c.id for c in onto.classes if c.l1 == "four_wheeler" and onto.is_thing(c.id)][:2]
    other = next(c.id for c in onto.classes if c.l1 == "vru" and onto.is_thing(c.id))
    classes = [*four, other]
    rng = np.random.default_rng(3)
    centres = {c: rng.normal(size=768) * 4 for c in classes}

    async def _go():
        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="PLANT-1", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(frames_per_track), city="BLR", sensors={},
                             ontology_version=onto.version))
            await db.flush()
            fids = []
            for f in range(frames_per_track):
                fid = uuid.uuid4()
                fids.append(fid)
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(f), cam_id="cam_f",
                             img_uri=f"s3://x/{f}.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            for c in classes:
                for t in range(n_tracks_per_class):
                    tid = uuid.uuid4()
                    db.add(Track(track_id=tid, session_id=sid, class_id=c, first_ts_ns=ts,
                                 last_ts_ns=ts + seconds_to_ns(frames_per_track)))
                    await db.flush()
                    for f, fid in enumerate(fids):
                        oid = uuid.uuid4()
                        x = 20.0 * t + f
                        db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=c,
                                      bbox=[x, 10.0, x + 40, 50.0], conf=0.9, source="human", state="accepted",
                                      attrs={}, provenance={}, version=1))
                        await db.flush()
                        vec = centres[c] + rng.normal(size=768) * 0.5
                        db.add(ObjectEmbedding(object_id=oid, dino_vec=vec.tolist(), model_versions={}))
            await db.commit()
        return str(sid)

    return _run(_go())


async def _classes(sid: str) -> list[tuple[str, int]]:
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Object.object_id, Object.class_id).join(Frame)
                                 .where(Frame.session_id == uuid.UUID(sid)).order_by(Object.object_id))).all()
    return [(str(o), int(c)) for o, c in rows]


def test_planted_errors_are_found_and_nothing_is_saved():
    from db.session import get_sessionmaker
    from services.errordetect.benchmark import planted_error_benchmark

    sid = _seed()
    before = _run(_classes(sid))

    async def _bench():
        async with get_sessionmaker()() as db:
            return await planted_error_benchmark(db, sid, rate=0.1, seed=1)

    out = _run(_bench())
    assert out["labels"] == 120 and out["planted"] == 12
    cl, tc = out["detectors"]["confident_learning"], out["detectors"]["track_consistency"]
    assert cl["recall"] >= 0.75, cl
    assert tc["recall"] >= 0.75, tc
    assert cl["precision"] and cl["precision"] >= 0.5
    # The whole run was rolled back.
    assert _run(_classes(sid)) == before


def test_the_benchmark_cannot_commit_and_restores_the_session_afterwards():
    from db.session import get_sessionmaker
    from services.errordetect import benchmark

    sid = _seed(n_tracks_per_class=3)

    async def _sneaky(db, session_id):
        await db.commit()  # a detector that tried to save would hit this
        return []

    async def _go():
        orig = benchmark._detectors
        benchmark._detectors = lambda: {"sneaky": _sneaky}
        try:
            async with get_sessionmaker()() as db:
                real = db.commit
                with pytest.raises(RuntimeError, match="never commits"):
                    await benchmark.planted_error_benchmark(db, sid)
                assert db.commit == real
        finally:
            benchmark._detectors = orig

    _run(_go())


def test_a_session_too_small_to_measure_is_refused():
    from db.session import get_sessionmaker
    from services.errordetect.benchmark import planted_error_benchmark

    sid = _seed(n_tracks_per_class=1, frames_per_track=3)

    async def _bench():
        async with get_sessionmaker()() as db:
            return await planted_error_benchmark(db, sid)

    out = _run(_bench())
    assert out["labels"] == 9 and "refused" in out


def test_the_endpoint_is_for_administrators():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    sid = _seed(n_tracks_per_class=4)
    with TestClient(app) as c:
        assert c.post("/api/errordetect/benchmark", json={"session_ids": [sid]},
                      headers=auth_headers("reviewer")).status_code == 403
        assert c.post("/api/errordetect/benchmark", json={"rate": 0.9}, headers=auth_headers("admin")).status_code == 400
        r = c.post("/api/errordetect/benchmark", json={"session_ids": [sid]}, headers=auth_headers("admin"))
        assert r.status_code == 200, r.text
        assert set(r.json()["pooled"]) == {"confident_learning", "embedding_outlier", "track_consistency"}
