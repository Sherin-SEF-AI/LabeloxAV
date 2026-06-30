"""M-L3.0: static scene elements (poles, road edges, buildings, vegetation, markings) are extracted from a
cloud, and extract_cloud geo-references them into static_element rows that feed the HD map as MapElement
candidates.

Extractor geometry needs no infra; extract_cloud needs DB + MinIO."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.lidar.extract import (
    extract_buildings,
    extract_markings,
    extract_poles,
    extract_road_edges,
)
from services.lidar.ingest import Cloud
from services.lidar.segment3d.semantic import road_class_id

PLANE = [0.0, 0.0, 1.0, 0.0]


def _ground(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(2, 30, n), rng.uniform(-8, 8, n), rng.normal(0, 0.02, n)], axis=1)


def _pole(x, y, h, n=250, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(x, 0.08, n), rng.normal(y, 0.08, n), rng.uniform(0.2, h, n)], axis=1)


def test_extract_poles():
    cloud = Cloud(xyz=np.vstack([_ground(), _pole(12, 3, 6.0)]).astype(np.float32),
                  intensity=np.ones(3250, np.float32), ts_ns=1, source="pseudo")
    poles = extract_poles(cloud, PLANE)
    assert len(poles) == 1
    assert poles[0]["kind"] == "pole" and poles[0]["pole_type"] == "street_light"
    assert 5.0 < poles[0]["height"] < 6.5


def test_extract_buildings():
    rng = np.random.default_rng(2)
    facade = np.stack([rng.uniform(5, 15, 400), rng.normal(8.0, 0.1, 400), rng.uniform(0.6, 8.0, 400)], axis=1)
    cloud = Cloud(xyz=np.vstack([_ground(), facade]).astype(np.float32),
                  intensity=np.ones(3400, np.float32), ts_ns=1, source="pseudo")
    buildings = extract_buildings(cloud, PLANE)
    assert len(buildings) >= 1
    assert buildings[0]["kind"] == "building" and buildings[0]["verticality"] > 0.8


def test_extract_road_edges_and_markings():
    onto = get_ontology()
    road_id = road_class_id(onto)
    rng = np.random.default_rng(3)
    n = 4000
    road = np.stack([rng.uniform(2, 30, n), rng.uniform(-4, 4, n), rng.normal(0, 0.02, n)], axis=1).astype(np.float32)
    # a bright lateral stop line at x=10, y in [-3,3]
    sl = np.stack([rng.normal(10, 0.1, 300), rng.uniform(-3, 3, 300), np.zeros(300)], axis=1).astype(np.float32)
    xyz = np.vstack([road, sl])
    inten = np.concatenate([rng.uniform(0.1, 0.3, n), np.full(300, 0.95, np.float32)]).astype(np.float32)
    semantic = np.full(len(xyz), road_id, dtype=np.int32)
    cloud = Cloud(xyz=xyz, intensity=inten, ts_ns=1, source="pseudo")

    edges = extract_road_edges(cloud, semantic, road_id, PLANE)
    assert {e["side"] for e in edges} == {"left", "right"} and all(len(e["line"]) >= 2 for e in edges)

    markings = extract_markings(cloud, semantic, road_id, PLANE)
    assert any(m["kind"] == "stop_line" for m in markings)


def _infra_up() -> bool:
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up")


def _clear():
    from db.session import get_engine, get_sessionmaker
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@requires_infra
def test_extract_cloud_feeds_hdmap():
    async def run():
        from geoalchemy2.elements import WKTElement

        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Frame, MapElement, StaticElement
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.extract import extract_cloud
        from services.lidar.ingest import store_cloud

        get_object_store().ensure_bucket()
        onto = get_ontology()
        sid, ts = uuid.uuid4(), now_ns()
        cloud = Cloud(xyz=np.vstack([_ground(), _pole(12, 3, 6.0)]).astype(np.float32),
                      intensity=np.ones(3250, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="EXT-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            db.add(Frame(session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x.jpg", width=1280,
                         height=960, quality=1.0, gnss=WKTElement("POINT(77.6 12.9)", srid=4326)))
            await db.commit()
        stored = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(stored["cloud_id"])

        res = await extract_cloud(cloud_id)
        assert res["geo_referenced"] and res["by_kind"].get("pole", 0) >= 1

        from sqlalchemy import func, select
        async with get_sessionmaker()() as db:
            n_static = (await db.execute(select(func.count()).select_from(StaticElement)
                        .where(StaticElement.session_id == sid))).scalar()
            n_map = (await db.execute(select(func.count()).select_from(MapElement)
                     .where(MapElement.source_sessions.any(str(sid))))).scalar()
            assert n_static >= 1 and n_map >= 1     # poles fed the HD map as MapElement candidates

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
