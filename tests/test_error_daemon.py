"""Corpus-wide error daemon: the consistency critic feeds the ErrorCandidate fix queue."""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


async def _seed_above_horizon_vehicle():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid, oid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(6).integers(30, 220, size=(1080, 1920, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_front/{ts}.jpg", buf.tobytes(), "image/jpeg")
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="ERR-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_front", img_uri=uri, width=1920, height=1080, quality=0.9))
        # a car whose box sits high in the frame -> its bottom is above the horizon -> geometric critic flag
        db.add(Object(object_id=oid, frame_id=fid, class_id=sedan, bbox=[850.0, 120.0, 1050.0, 300.0], conf=0.9,
                      source="fused", state="review", provenance={}, attrs={}, version=1))
        await db.commit()
    return str(sid), str(oid)


@requires_infra
def test_critic_detector_feeds_fix_queue():
    from db.models import ErrorCandidate
    from db.session import get_sessionmaker
    from services.errordetect.critic_detector import detect_critic
    from services.errordetect.queue import run_detection
    from sqlalchemy import select

    sid, oid = run_async(_seed_above_horizon_vehicle())

    async def _flow():
        async with get_sessionmaker()() as db:
            found = await detect_critic(db, sid)
            assert any(c["object_id"] == oid and "geometric" in c["detail"]["checks"] for c in found)
        # run_detection persists it as a ranked ErrorCandidate
        async with get_sessionmaker()() as db:
            res = await run_detection(db, sid, kinds=["critic_flag"])
            assert res["persisted"] >= 1
        async with get_sessionmaker()() as db:
            cands = (await db.execute(select(ErrorCandidate).where(ErrorCandidate.kind == "critic_flag"))).scalars().all()
            assert any(str(c.object_id) == oid for c in cands)

    run_async(_flow())
