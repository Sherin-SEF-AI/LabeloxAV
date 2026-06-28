"""M-L3.3: the 3D quality checker catches floating, below-ground, impossibly-sized, duplicate, and misaligned
boxes and missing neighbours, 3D scene classification reads structure (tunnel vs open), 3D rare mining
surfaces flooded roads and animals, and a confirmed flag demotes the object back to review.

Checks and classifiers need no infra; check_cloud + confirm need DB + MinIO."""

from __future__ import annotations

import asyncio
import math
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.lidar.ingest import Cloud
from services.lidar.quality3d import (
    check_cuboid,
    classify_3d_structure,
    find_missing_neighbors,
    mine_3d_cues,
)
from services.lidar.segment3d.semantic import road_class_id

PLANE = [0.0, 0.0, 1.0, 0.0]


def _box_points(center, dims, yaw=0.0, n=600, seed=0):
    rng = np.random.default_rng(seed)
    local = rng.uniform(-0.5, 0.5, (n, 3)) * np.array(dims)
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (local @ r.T + np.array(center)).astype(np.float32)


def test_quality_checker_catches_defects():
    car = _box_points([10, 0, 0.75], [4, 1.8, 1.5])
    cloud_xyz = np.vstack([car, np.random.default_rng(1).uniform([-5, -5, -0.02], [25, 5, 0.02], (2000, 3))]).astype(np.float32)

    good = {"center": [10, 0, 0.75], "dims": [4, 1.8, 1.5], "yaw": 0.0}
    assert check_cuboid(good, cloud_xyz, PLANE) == []

    floating = {"center": [10, 0, 3.0], "dims": [4, 1.8, 1.5], "yaw": 0.0}
    assert any(f["kind"] == "floating" for f in check_cuboid(floating, cloud_xyz, PLANE))

    below = {"center": [10, 0, -1.0], "dims": [4, 1.8, 1.5], "yaw": 0.0}
    assert any(f["kind"] == "below_ground" for f in check_cuboid(below, cloud_xyz, PLANE))

    huge = {"center": [10, 0, 0.75], "dims": [50, 2, 2], "yaw": 0.0}
    assert any(f["kind"] == "impossible_dims" for f in check_cuboid(huge, cloud_xyz, PLANE))

    empty = {"center": [40, 20, 0.75], "dims": [4, 1.8, 1.5], "yaw": 0.0}
    assert any(f["kind"] == "misaligned" for f in check_cuboid(empty, cloud_xyz, PLANE))

    dup_flags = check_cuboid(good, cloud_xyz, PLANE, neighbors=[{"object_3d_id": "x", **good}])
    assert any(f["kind"] == "duplicate" for f in dup_flags)


def test_missing_neighbor():
    car = _box_points([14, -3, 0.9], [4, 1.8, 1.8], n=400)   # a dense cluster, no cuboid covers it
    cloud_xyz = np.vstack([car, np.random.default_rng(2).uniform([-5, -5, -0.02], [25, 5, 0.02], (2000, 3))]).astype(np.float32)
    missing = find_missing_neighbors(cloud_xyz, PLANE, cuboids=[])
    assert any(m["kind"] == "missing_neighbor" for m in missing)


def test_3d_structure_tunnel_vs_open():
    rng = np.random.default_rng(3)
    ground = np.stack([rng.uniform(2, 30, 3000), rng.uniform(-6, 6, 3000), rng.normal(0, 0.02, 3000)], axis=1)
    roof = np.stack([rng.uniform(3, 25, 1000), rng.uniform(-3, 3, 1000), rng.uniform(4.5, 5.0, 1000)], axis=1)
    wl = np.stack([rng.uniform(3, 25, 600), rng.normal(4.0, 0.1, 600), rng.uniform(1, 4, 600)], axis=1)
    wr = np.stack([rng.uniform(3, 25, 600), rng.normal(-4.0, 0.1, 600), rng.uniform(1, 4, 600)], axis=1)
    tunnel = Cloud(xyz=np.vstack([ground, roof, wl, wr]).astype(np.float32),
                   intensity=np.ones(5200, np.float32), ts_ns=1, source="pseudo")
    assert classify_3d_structure(tunnel, PLANE, 2)["3d_structure"] == "tunnel"
    open_cloud = Cloud(xyz=ground.astype(np.float32), intensity=np.ones(3000, np.float32), ts_ns=1, source="pseudo")
    assert classify_3d_structure(open_cloud, PLANE, 2)["3d_structure"] == "open"


def test_3d_rare_cues():
    onto = get_ontology()
    road_id = road_class_id(onto)
    rng = np.random.default_rng(4)
    road = np.stack([rng.uniform(2, 30, 1000), rng.uniform(-4, 4, 1000), rng.normal(0, 0.01, 1000)], axis=1).astype(np.float32)
    sem = np.full(1000, road_id, np.int32)
    flooded = Cloud(xyz=road, intensity=np.full(1000, 0.04, np.float32), ts_ns=1, source="pseudo")
    cues = mine_3d_cues(flooded, sem, PLANE, [], road_id, onto)
    assert any(c["kind"] == "3d_flooded_road" for c in cues)

    cattle = onto.by_name("cattle").id
    cubs = [{"class_id": cattle, "center": [10, 0, 0.75], "dims": [1.8, 0.8, 1.4], "yaw": 0.0}]
    animal_cues = mine_3d_cues(flooded, sem, PLANE, cubs, road_id, onto)
    assert any(c["kind"] == "3d_animal_crossing" for c in animal_cues)


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
def test_check_cloud_flags_and_confirm_demotes():
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Object3D, QualityFlag3D
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.ingest import store_cloud
        from services.lidar.quality3d import check_cloud, confirm_flag

        get_object_store().ensure_bucket()
        onto = get_ontology()
        sedan = onto.by_name("sedan").id
        sid, ts = uuid.uuid4(), now_ns()
        rng = np.random.default_rng(0)
        ground = np.stack([rng.uniform(2, 30, 3000), rng.uniform(-8, 8, 3000), rng.normal(0, 0.02, 3000)], axis=1)
        cloud = Cloud(xyz=ground.astype(np.float32), intensity=np.ones(3000, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="QC-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.commit()
        stored = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(stored["cloud_id"])
        async with get_sessionmaker()() as db:
            # a floating box high above the ground with no enclosed points
            o = Object3D(cloud_id=cloud_id, class_id=sedan, center=[12, 0, 4.0], dims=[4, 1.8, 1.5], yaw=0.0,
                         conf=0.8, box_source="lifted", source="fused", state="auto_accept")
            db.add(o)
            await db.flush()
            oid = o.object_3d_id
            await db.commit()

        res = await check_cloud(cloud_id)
        assert res["flags"] >= 1
        from sqlalchemy import select
        async with get_sessionmaker()() as db:
            flags = (await db.execute(select(QualityFlag3D).where(QualityFlag3D.object_3d_id == oid))).scalars().all()
            assert any(f.kind == "floating" for f in flags)
            fid = flags[0].flag_id

        await confirm_flag(fid)
        async with get_sessionmaker()() as db:
            o2 = await db.get(Object3D, oid)
            assert o2.state == "review"            # the confirmed flag demoted the object into the review loop

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
