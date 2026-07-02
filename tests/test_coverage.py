"""Coverage-gap analyzer: reports class balance, scene-axis coverage, and named gaps."""

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


async def _seed_scene_frame():
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(5).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CV-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=640, height=480,
                     quality=0.9, scene={"weather": "clear", "time_of_day": "day", "road_type": "urban", "density": "moderate"}))
        await db.commit()
    return str(sid)


@requires_infra
def test_coverage_report_structure_and_gaps():
    from db.session import get_sessionmaker
    from services.agent.coverage import analyze_coverage

    run_async(_seed_scene_frame())

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await analyze_coverage(db)
        assert set(r["scene_coverage"].keys()) == {"weather", "time_of_day", "road_type", "density"}
        assert r["scene_frames"] >= 1
        assert isinstance(r["gaps"], list) and r["gaps"]      # a fresh corpus always has gaps
        assert "missing" in r["class_balance"] and "rare" in r["class_balance"]
        # the seeded frame only has clear/day; some other weather value should read as a gap
        assert any("weather=" in g for g in r["gaps"])

    run_async(_flow())
