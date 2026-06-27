"""P2 analytics + eval dashboard tests (the sales sheet). Requires infra (DB).

Sync tests: a TestClient drives a tiny ASGI app that mounts ONLY analytics.router (so the suite
passes before main.py registers it), while seeding and direct-call assertions use asyncio.run.
The app caches one engine per process, so we clear that cache around each loop boundary (run_async)
to avoid binding an engine to a closed/foreign loop. Mirrors tests/test_m6_api.py.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from geoalchemy2 import WKTElement

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


# Class ids used for seeding: autorickshaw (6, india) and sedan (11).
CLS_AUTO = 6
CLS_SEDAN = 11


async def _seed_coro():
    from db.models import Frame, Object, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()

    async with maker() as db:
        db.add(
            DbSession(
                session_id=sid,
                vehicle_id="TIGOR-09",
                start_ts_ns=start,
                end_ts_ns=start + seconds_to_ns(1),
                city="BLR",
                sensors={},
                ontology_version="labelox-in-0.1.0",
            )
        )
        db.add(
            Frame(
                frame_id=fid,
                session_id=sid,
                ts_ns=start,
                cam_id="cam_f",
                img_uri="s3://x",
                width=640,
                height=480,
                quality=0.9,
                gnss=WKTElement("POINT(77.5946 12.9716)", srid=4326),
            )
        )
        # Two auto-accepted autorickshaws, one human-accepted autorickshaw, one human-accepted sedan.
        oid_accept_confirm = uuid.uuid4()
        oid_accept_reclass = uuid.uuid4()
        db.add(
            Object(object_id=uuid.uuid4(), frame_id=fid, class_id=CLS_AUTO, bbox=[1, 1, 2, 2],
                   conf=0.97, attrs={}, source="auto_accept", state="auto_accept")
        )
        db.add(
            Object(object_id=uuid.uuid4(), frame_id=fid, class_id=CLS_AUTO, bbox=[3, 3, 4, 4],
                   conf=0.96, attrs={}, source="auto_accept", state="auto_accept")
        )
        db.add(
            Object(object_id=oid_accept_confirm, frame_id=fid, class_id=CLS_AUTO, bbox=[5, 5, 6, 6],
                   conf=0.7, attrs={}, source="human", state="accepted")
        )
        db.add(
            Object(object_id=oid_accept_reclass, frame_id=fid, class_id=CLS_SEDAN, bbox=[7, 7, 8, 8],
                   conf=0.5, attrs={}, source="human", state="accepted")
        )
        await db.flush()  # objects must exist before review rows (FK on object_id)
        # One review confirms the auto class (before == after), one reclassifies it (before != after).
        db.add(
            Review(review_id=uuid.uuid4(), object_id=oid_accept_confirm, reviewer="sherin",
                   action="confirm", before={"class_id": CLS_AUTO}, after={"class_id": CLS_AUTO},
                   time_spent_ms=1000, ts_ns=start)
        )
        db.add(
            Review(review_id=uuid.uuid4(), object_id=oid_accept_reclass, reviewer="sherin",
                   action="reclassify", before={"class_id": CLS_AUTO}, after={"class_id": CLS_SEDAN},
                   time_spent_ms=3000, ts_ns=start)
        )
        await db.commit()
    return str(sid)


def _seed():
    return run_async(_seed_coro())


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from services.api.routers import analytics

    # Tiny app that mounts ONLY the analytics router, so the suite is green before integration.
    app = FastAPI()
    app.include_router(analytics.router, prefix="/api")
    _clear_db_cache()
    return TestClient(app)


@requires_infra
def test_class_distribution_counts_and_long_tail():
    from services.analytics.dashboards import class_distribution

    sid = _seed()
    rows = run_async(class_distribution(sid))
    by_id = {r["class_id"]: r for r in rows}
    assert by_id[CLS_AUTO]["count"] == 3
    assert by_id[CLS_SEDAN]["count"] == 1
    assert by_id[CLS_AUTO]["india"] is True
    # Sorted desc and only the seeded classes have objects in this session.
    assert rows == sorted(rows, key=lambda r: r["count"], reverse=True)
    assert sum(r["count"] for r in rows) == 4


@requires_infra
def test_label_source_mix_percentages_present():
    from services.analytics.dashboards import label_source_mix

    sid = _seed()
    mix = run_async(label_source_mix(sid))
    assert mix["total"] == 4
    assert mix["by_state"]["auto_accept"] == 2
    assert mix["by_state"]["accepted"] == 2
    assert mix["by_source"]["auto_accept"] == 2
    assert mix["by_source"]["human"] == 2
    assert mix["auto_accepted_pct"] == 50.0
    assert mix["human_touched_pct"] == 50.0


@requires_infra
def test_review_agreement_confirmed_vs_reclassified():
    from services.analytics.dashboards import review_agreement

    _seed()
    agr = run_async(review_agreement())
    # Global counts include prior rows; assert the relationship holds and our two are reflected.
    assert agr["total_reviews"] >= 2
    assert agr["confirmed"] + agr["reclassified"] == agr["total_reviews"]
    assert agr["confirmed"] >= 1
    assert agr["reclassified"] >= 1
    assert 0.0 <= agr["confirmed_pct"] <= 100.0
    assert agr["mean_time_spent_ms"] > 0


@requires_infra
def test_overview_rollup():
    from services.analytics.dashboards import overview

    from services.autolabel.ontology import get_ontology

    sid = _seed()
    ov = run_async(overview(sid))
    assert ov["sessions"] == 1
    assert ov["frames"] == 1
    assert ov["objects"] == 4
    assert ov["long_tail"]["total_classes"] == len(get_ontology().classes)
    assert ov["long_tail"]["covered_classes"] == 2
    assert "auto_accepted_pct" in ov
    assert "source_mix" in ov


@requires_infra
def test_geo_points_returns_lat_lon():
    from services.analytics.dashboards import geo_points

    sid = _seed()
    pts = run_async(geo_points(sid))
    assert len(pts) == 1
    assert round(pts[0]["lat"], 3) == 12.972
    assert round(pts[0]["lon"], 3) == 77.595


@requires_infra
def test_api_endpoints_via_testclient():
    sid = _seed()
    with _client() as c:
        ov = c.get(f"/api/analytics/overview?session_id={sid}").json()
        assert ov["objects"] == 4

        classes = c.get(f"/api/analytics/classes?session_id={sid}").json()
        assert any(r["class_id"] == CLS_AUTO and r["count"] == 3 for r in classes)

        mix = c.get(f"/api/analytics/source-mix?session_id={sid}").json()
        assert mix["total"] == 4

        scen = c.get(f"/api/analytics/scenarios?session_id={sid}").json()
        assert isinstance(scen, list)

        geo = c.get(f"/api/analytics/geo?session_id={sid}").json()
        assert len(geo) == 1

        agr = c.get("/api/analytics/review-agreement").json()
        assert agr["total_reviews"] >= 2
