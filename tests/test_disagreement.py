"""Champion-vs-challenger disagreement mining from recorded fusion provenance."""

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


def test_disagreement_pure_function():
    from services.agent.disagreement import _disagreement

    class _O:
        class_id = -1  # unknown -> accepted resolves to None, dissent = max conf
        provenance = {"agreement": False, "proposals": [
            {"class_name": "sedan", "conf": 0.8, "path": "path_a_yolo26"},
            {"class_name": "suv", "conf": 0.6, "path": "path_b_sam3"},
        ]}

    d = _disagreement(_O())
    assert d is not None
    strength, classes = d
    assert strength >= 0.6 and classes == ["sedan", "suv"]

    class _Agree:
        class_id = -1
        provenance = {"proposals": [{"class_name": "sedan", "conf": 0.8}, {"class_name": "sedan", "conf": 0.7}]}

    assert _disagreement(_Agree()) is None   # both paths voted the same class


async def _seed_disagreeing_object():
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
    img = np.random.default_rng(7).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    sedan = next(c.id for c in onto.classes if c.name == "sedan")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="DG-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.9))
        db.add(Object(object_id=oid, frame_id=fid, class_id=sedan, bbox=[10.0, 10.0, 60.0, 90.0], conf=0.7,
                      source="fused", state="review", attrs={},
                      provenance={"agreement": False, "proposals": [
                          {"class_name": "sedan", "conf": 0.7, "path": "path_a_yolo26"},
                          {"class_name": "suv", "conf": 0.55, "path": "path_b_sam3"}]}, version=1))
        await db.commit()
    return str(sid)


@requires_infra
def test_mine_disagreements_writes_scenario():
    from db.models import ScenarioCandidate
    from db.session import get_sessionmaker
    from services.agent.disagreement import mine_disagreements
    from sqlalchemy import select

    sid = run_async(_seed_disagreeing_object())

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await mine_disagreements(db, sid)
            assert r["persisted"] >= 1
        async with get_sessionmaker()() as db:
            cands = (await db.execute(select(ScenarioCandidate).where(
                ScenarioCandidate.session_id == uuid.UUID(sid), ScenarioCandidate.kind == "model_disagreement"))).scalars().all()
            assert cands and "suv" in cands[0].tag

    run_async(_flow())
