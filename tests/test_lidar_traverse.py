"""M-L3.1: a 3D free-space grid and a metric drivable grid are produced, the road surface is classified
(including water and unpaved), and the elevation profile identifies a ramp or flyover.

Grid/surface/elevation need no infra; traverse_cloud needs DB + MinIO."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.lidar.ingest import Cloud
from services.lidar.segment3d.semantic import road_class_id
from services.lidar.traverse import (
    classify_surface,
    drivable_grid,
    elevation_profile,
    freespace_grid,
)

PLANE = [0.0, 0.0, 1.0, 0.0]


def _ground(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(2, 40, n), rng.uniform(-10, 10, n), rng.normal(0, 0.02, n)], axis=1)


def _obstacle(x, y, h, n=400, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.normal(x, 0.5, n), rng.normal(y, 0.5, n), rng.uniform(0.4, h, n)], axis=1)


def test_freespace_grid():
    cloud = Cloud(xyz=np.vstack([_ground(), _obstacle(15, 0, 3.0)]).astype(np.float32),
                  intensity=np.ones(4400, np.float32), ts_ns=1, source="pseudo")
    fs = freespace_grid(cloud, PLANE)
    assert fs["occupied_cells"] > 0 and fs["free_cells"] > 0 and fs["free_frac"] < 1.0


def test_drivable_grid():
    onto = get_ontology()
    road_id = road_class_id(onto)
    ground = _ground()
    obstacle = _obstacle(15, 0, 3.0)
    xyz = np.vstack([ground, obstacle]).astype(np.float32)
    semantic = np.concatenate([np.full(len(ground), road_id), np.full(len(obstacle), -1)]).astype(np.int32)
    cloud = Cloud(xyz=xyz, intensity=np.ones(len(xyz), np.float32), ts_ns=1, source="pseudo")
    dr = drivable_grid(cloud, semantic, road_id, PLANE)
    assert dr["drivable_cells"] > 0 and dr["non_drivable_cells"] > 0


def test_surface_classification():
    onto = get_ontology()
    road_id = road_class_id(onto)
    rng = np.random.default_rng(2)
    flat = np.stack([rng.uniform(2, 30, 1000), rng.uniform(-4, 4, 1000), rng.normal(0, 0.01, 1000)], axis=1).astype(np.float32)
    sem = np.full(1000, road_id, np.int32)
    water = Cloud(xyz=flat, intensity=np.full(1000, 0.04, np.float32), ts_ns=1, source="pseudo")
    assert classify_surface(water, sem, road_id, PLANE)["surface"] == "water"
    asphalt = Cloud(xyz=flat, intensity=rng.uniform(0.3, 0.5, 1000).astype(np.float32), ts_ns=1, source="pseudo")
    assert classify_surface(asphalt, sem, road_id, PLANE)["surface"] == "asphalt"
    rough = np.stack([rng.uniform(2, 30, 1000), rng.uniform(-4, 4, 1000), rng.normal(0, 0.12, 1000)], axis=1).astype(np.float32)
    gravel = Cloud(xyz=rough, intensity=rng.uniform(0.35, 0.5, 1000).astype(np.float32), ts_ns=1, source="pseudo")
    assert classify_surface(gravel, sem, road_id, PLANE)["surface"] in ("gravel", "mud")


def test_elevation_detects_ramp():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 60, 4000)
    z = 0.05 * x + rng.normal(0, 0.02, 4000)               # a steady 5 percent climb
    ramp = np.stack([x, rng.uniform(-4, 4, 4000), z], axis=1).astype(np.float32)
    cloud = Cloud(xyz=ramp, intensity=np.ones(4000, np.float32), ts_ns=1, source="pseudo")
    prof = elevation_profile(cloud, PLANE)
    assert prof["feature"] in ("ramp", "flyover", "incline") and prof["max_slope"] > 0.02


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
def test_traverse_cloud_persists():
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Session as DbSession
        from db.models import Traversability
        from db.session import get_sessionmaker
        from services.lidar.ingest import store_cloud
        from services.lidar.traverse import traverse_cloud

        get_object_store().ensure_bucket()
        sid, ts = uuid.uuid4(), now_ns()
        cloud = Cloud(xyz=np.vstack([_ground(), _obstacle(15, 0, 3.0)]).astype(np.float32),
                      intensity=np.ones(4400, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="TRV-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=get_ontology().version))
            await db.commit()
        stored = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(stored["cloud_id"])

        res = await traverse_cloud(cloud_id)
        assert res["occupied_cells"] > 0 and 0.0 <= res["free_frac"] <= 1.0
        from sqlalchemy import select
        async with get_sessionmaker()() as db:
            row = (await db.execute(select(Traversability).where(Traversability.cloud_id == cloud_id))
                   ).scalar_one()
            assert row.freespace_uri and row.drivable_uri and row.surface_class

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
