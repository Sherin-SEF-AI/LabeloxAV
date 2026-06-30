"""M-L2.3: per-point segmentation labels points from the 3D cuboids and the ground using the ontology, gives
each cuboid its own instance, flags low-confidence (unlabeled) regions, and the PTv3 path parks on the seam.

Geometry and projection need no infra; segment_cloud needs DB + MinIO."""

from __future__ import annotations

import asyncio
import math
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.lidar.ingest import Cloud
from services.lidar.segment3d import (
    SegmentationUnavailable,
    points_in_cuboid,
    road_class_id,
    segment_projected,
    segment_ptv3,
)


def _box_points(center, dims, yaw=0.0, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    local = rng.uniform(-0.5, 0.5, (n, 3)) * np.array(dims)
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (local @ r.T + np.array(center)).astype(np.float32)


def test_points_in_cuboid():
    cub = {"center": [10, 0, 1], "dims": [4, 2, 2], "yaw": 0.0}
    inside = np.array([[10, 0, 1], [11.5, 0.5, 1.5]], dtype=np.float32)
    outside = np.array([[20, 0, 1], [10, 5, 1]], dtype=np.float32)
    assert points_in_cuboid(inside, cub).all()
    assert not points_in_cuboid(outside, cub).any()


def test_segment_projected_from_cuboids_and_ground():
    onto = get_ontology()
    sedan = onto.by_name("sedan").id
    rng = np.random.default_rng(1)
    ground = np.stack([rng.uniform(2, 30, 2000), rng.uniform(-8, 8, 2000), rng.normal(0, 0.02, 2000)], axis=1)
    car = _box_points([12, 0, 0.75], [4, 1.8, 1.5], n=1000)
    bg = np.stack([rng.uniform(35, 45, 500), rng.uniform(10, 20, 500), rng.uniform(3, 6, 500)], axis=1)
    cloud = Cloud(xyz=np.vstack([ground, car, bg]).astype(np.float32),
                  intensity=np.ones(3500, np.float32), ts_ns=1, source="pseudo")
    cub = {"center": [12, 0, 0.75], "dims": [4, 1.8, 1.5], "yaw": 0.0, "class_id": sedan}
    res = segment_projected(cloud, [cub], [0.0, 0.0, 1.0, 0.0])

    sem, inst = res["semantic"], res["instance"]
    car_sem = sem[2000:3000]
    assert (car_sem == sedan).mean() > 0.9                  # the car cluster takes the sedan class
    assert (inst[2000:3000] == 0).mean() > 0.9              # and instance 0
    assert (sem[:2000] == road_class_id(onto)).mean() > 0.8  # ground is road
    assert (sem[3000:] == -1).all()                         # far background is unlabeled
    assert res["low_conf_frac"] > 0 and sedan in res["classes_present"]


def test_ptv3_parks_on_seam():
    cloud = Cloud(xyz=np.zeros((5, 3), np.float32), intensity=np.zeros(5, np.float32), ts_ns=1, source="lidar")
    with pytest.raises(SegmentationUnavailable):
        segment_ptv3(cloud)


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
def test_segment_cloud_persists_and_flags():
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Object3D, PointSegmentation
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.ingest import store_cloud
        from services.lidar.segment3d import segment_cloud

        get_object_store().ensure_bucket()
        onto = get_ontology()
        sedan = onto.by_name("sedan").id
        sid, ts = uuid.uuid4(), now_ns()
        rng = np.random.default_rng(0)
        ground = np.stack([rng.uniform(2, 30, 2000), rng.uniform(-8, 8, 2000), rng.normal(0, 0.02, 2000)], axis=1)
        car = _box_points([12, 0, 0.75], [4, 1.8, 1.5], n=1000)
        cloud = Cloud(xyz=np.vstack([ground, car]).astype(np.float32),
                      intensity=np.ones(3000, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="SEG-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.commit()
        res_store = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(res_store["cloud_id"])
        async with get_sessionmaker()() as db:
            db.add(Object3D(cloud_id=cloud_id, class_id=sedan, center=[12, 0, 0.75], dims=[4, 1.8, 1.5],
                            yaw=0.0, conf=0.9, box_source="lifted", source="fused", state="auto_accept"))
            await db.commit()

        res = await segment_cloud(cloud_id)
        assert res["method"] == "projected_2d" and res["kind"] == "panoptic"
        assert sedan in res["classes_present"] and 0.0 <= res["low_conf_frac"] <= 1.0
        assert res["n_instances"] == 1

        from sqlalchemy import select
        async with get_sessionmaker()() as db:
            row = (await db.execute(select(PointSegmentation).where(PointSegmentation.cloud_id == cloud_id))
                   ).scalar_one()
            assert row.labels_uri and row.n_points == 3000

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
