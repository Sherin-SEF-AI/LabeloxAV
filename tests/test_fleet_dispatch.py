"""Fleet Dispatch: gap parsing + priority (pure) and end-to-end order generation from controlled gaps with
the dispatch status flow."""

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


def test_parse_gap_and_priority():
    from services.agent.fleet_dispatch import _parse_gap, _priority, _window_for

    assert _parse_gap("weather=rain thin (26 frames, 0.7%)") == ("weather", "rain", "rain conditions")
    assert _parse_gap("time_of_day=night thin (5 frames)") == ("time_of_day", "night", "night driving")
    assert _parse_gap("129 ontology classes have no labels") is None      # not a driving-collectable gap
    assert _window_for("time_of_day", "night") == "18:00-22:00"
    # rain gap with a matching rain forecast outranks the same gap with no forecast
    assert _priority("weather", "rain", "rain") > _priority("weather", "rain", "unknown")


async def _seed_session():
    from db.models import OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    maker = get_sessionmaker()
    ts = now_ns()
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        db.add(DbSession(session_id=uuid.uuid4(), vehicle_id="FLEET-7", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={}, ontology_version=onto.version))
        await db.commit()


@requires_infra
def test_plan_orders_and_dispatch():
    import services.agent.coverage as coverage
    from db.models import CollectionOrder
    from db.session import get_sessionmaker
    from services.agent.fleet_dispatch import list_orders, plan_collection, set_status

    run_async(_seed_session())

    async def _fake_coverage(db, **kw):
        return {"gaps": ["weather=rain thin (26 frames, 0.7%)", "time_of_day=night thin (5 frames)"]}

    _orig = coverage.analyze_coverage
    coverage.analyze_coverage = _fake_coverage

    async def _flow():
        async with get_sessionmaker()() as db:
            res = await plan_collection(db)
        assert res["orders"] == 2 and res["vehicles"] >= 1
        async with get_sessionmaker()() as db:
            orders = await list_orders(db, "proposed")
        assert len(orders) == 2
        assert orders[0]["gap_kind"] == "weather"           # rain (0.8) outranks night (0.7)
        assert "starved of" in orders[0]["summary"]
        oid = orders[0]["order_id"]
        async with get_sessionmaker()() as db:
            await set_status(db, uuid.UUID(oid), "dispatched")
        async with get_sessionmaker()() as db:
            o = await db.get(CollectionOrder, uuid.UUID(oid))
            assert o.status == "dispatched"

    try:
        run_async(_flow())
    finally:
        coverage.analyze_coverage = _orig   # do not leak the stub into sibling tests
