"""Fleet-dispatch from flywheel tests: the cell classifier splits scene cells from class cells, and
plan_from_flywheel turns a cycle's collection tasks into ranked collection orders (windowed scene drives +
class-collection orders with target counts), defaulting to the fleet's dominant city."""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns
from services.agent.fleet_dispatch import _classify_cell, list_orders, plan_from_flywheel

pytestmark = pytest.mark.db


def test_classify_cell_scene_vs_class():
    assert _classify_cell("night_rain") == ("scene", "night", "rain")
    assert _classify_cell("day_clear") == ("scene", "day", "clear")
    assert _classify_cell("elephant") == ("class", "elephant", None)
    # a class whose name contains an underscore but is not a scene cell stays a class
    assert _classify_cell("goat_herd") == ("class", "goat_herd", None)


@pytest.mark.asyncio
async def test_plan_from_flywheel_makes_scene_and_class_orders():
    from db.models import FlywheelCycle, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    maker = get_sessionmaker()
    ts = now_ns()
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        # a fleet that mostly drives BLR, plus one stray city
        db.add(DbSession(session_id=uuid.uuid4(), vehicle_id="TIGOR-07", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={}, ontology_version=onto.version))
        db.add(DbSession(session_id=uuid.uuid4(), vehicle_id="TIGOR-07", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={}, ontology_version=onto.version))
        db.add(DbSession(session_id=uuid.uuid4(), vehicle_id="RT-01", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="DEL", sensors={}, ontology_version=onto.version))
        cycle = FlywheelCycle(
            label_budget=2000,
            signals={"source": "corpus"},
            allocation=[],
            collection_tasks=[
                {"cell": "night_rain", "priority": 0.18, "target_count": 3612, "missing": False},
                {"cell": "day_fog", "priority": 0.03, "target_count": 1235, "missing": True},
                {"cell": "elephant", "priority": 0.003, "target_count": 121, "missing": True},
            ],
            rationale="test")
        db.add(cycle)
        await db.commit()
        cid = cycle.cycle_id

    async with maker() as db:
        res = await plan_from_flywheel(db, cycle_id=cid, created_by="test_fw")
        assert res["orders"] == 3 and res["scene"] == 2 and res["classes"] == 1

    async with maker() as db:
        rows = await list_orders(db, status="proposed")
        targets = [o["target"] for o in rows]
        assert any("night rain driving (need 3,612)" in t for t in targets)   # scene order, with count
        assert any("elephant sightings (need 121)" in t for t in targets)     # class-collection order
        assert any("day fog driving (need 1,235)" in t for t in targets)      # scene order (missing cell)
