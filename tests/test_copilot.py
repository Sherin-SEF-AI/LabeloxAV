"""Conversational corpus copilot: parse a plain-language question into facets and query matching frames."""

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


def test_parse_query_facets():
    from services.agent.copilot import parse_query
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    f = parse_query("two-wheelers going against traffic at night on the highway", onto)
    assert f["scene"] == {"time_of_day": "night", "road_type": "highway"}
    assert f["attrs"].get("direction") == "wrong_way"     # 'against traffic' -> wrong_way
    assert len(f["class_ids"]) >= 5                        # two-wheeler classes
    assert parse_query("near-miss with a pedestrian", onto)["safety"] is True


async def _seed_night_pedestrian():
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    ts = now_ns()
    img = np.random.default_rng(4).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=640, height=480,
                     quality=0.9, scene={"time_of_day": "night", "weather": "clear", "road_type": "urban", "density": "sparse"}))
        db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=ped, bbox=[10.0, 10.0, 40.0, 90.0], conf=0.9,
                      source="fused", state="review", attrs={"direction": "cross"}, provenance={}, version=1))
        await db.commit()
    return str(fid)


@requires_infra
def test_answer_finds_matching_frame():
    from db.session import get_sessionmaker
    from services.agent.copilot import answer_corpus_query

    fid = run_async(_seed_night_pedestrian())

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await answer_corpus_query(db, "pedestrians crossing at night")
        ids = {f["frame_id"] for f in r["frames"]}
        assert fid in ids and r["count"] >= 1
        assert "night" in r["understood"]

    run_async(_flow())


@requires_infra
def test_dataset_report_structure():
    from db.session import get_sessionmaker
    from services.agent.copilot import dataset_report

    run_async(_seed_night_pedestrian())

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await dataset_report(db)
        assert set(r["size"].keys()) == {"sessions", "objects", "human_labeled"}
        assert r["size"]["objects"] >= 1
        assert isinstance(r["coverage_gaps"], list)
        assert "fix_queue_total" in r and "scenarios" in r

    run_async(_flow())
