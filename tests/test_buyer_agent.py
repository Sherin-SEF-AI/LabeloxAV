"""Buyer Curation Agent: target-count parsing (pure) and honest availability/shortfall over seeded frames."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
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


def test_target_count_parsing():
    from services.agent.buyer_agent import _target_count

    assert _target_count("10k frames with a pedestrian") == 10000
    assert _target_count("give me 1,200 night frames") == 1200
    assert _target_count("500 frames") == 500
    assert _target_count("near-miss under 2.5s ttc") is None   # small incidental numbers ignored
    assert _target_count("all pedestrians") is None


async def _seed_ped_frames(n: int = 5):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    maker = get_sessionmaker()
    ts = now_ns()
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        sid = uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id="BUYER-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        for i in range(n):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f", img_uri=f"s3://x/{i}.jpg",
                         width=320, height=240, quality=0.9, scene={"time_of_day": "day", "weather": "clear"}))
            await db.flush()
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=ped, bbox=[1.0, 1.0, 30.0, 90.0],
                          conf=0.9, source="fused", state="accepted", attrs={}, provenance={}, version=1))
        await db.commit()
    return sid


@requires_infra
def test_analyze_spec_reports_availability_and_shortfall():
    from db.session import get_sessionmaker
    from services.agent.buyer_agent import analyze_spec

    run_async(_seed_ped_frames(5))

    async def _flow():
        async with get_sessionmaker()() as db:
            fit = await analyze_spec(db, "3 frames with a pedestrian")
            short = await analyze_spec(db, "1000 frames with a pedestrian")
        assert "pedestrian" in fit["understood"]
        assert fit["fulfillment"]["available"] >= 5
        assert fit["fulfillment"]["fulfillable"] == 3 and fit["fulfillment"]["shortfall"] == 0
        assert short["fulfillment"]["shortfall"] >= 900 and short["guidance"]

    run_async(_flow())
