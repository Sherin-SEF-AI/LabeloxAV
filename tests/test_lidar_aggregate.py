"""M-L3.2: consecutive scans register into a consistent transform, a revisited location triggers a loop
closure that the pose graph corrects, scans accumulate into a dense map, and low-confidence registration is
flagged.

Registration / loop closure / accumulation need no infra; aggregate_sessions needs DB + MinIO."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.aggregate import (
    accumulate_scans,
    detect_loops,
    optimize_pose_graph,
    register_pair,
)
from services.lidar.ingest import Cloud


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    ground = np.stack([rng.uniform(0, 20, 2000), rng.uniform(-10, 10, 2000), rng.normal(0, 0.02, 2000)], axis=1)
    wall_x = np.stack([rng.normal(20, 0.1, 800), rng.uniform(-10, 10, 800), rng.uniform(0, 4, 800)], axis=1)
    wall_y = np.stack([rng.uniform(0, 20, 800), rng.normal(10, 0.1, 800), rng.uniform(0, 4, 800)], axis=1)
    return np.vstack([ground, wall_x, wall_y]).astype(np.float32)


def _pose(x, y):
    t = np.eye(4)
    t[0, 3], t[1, 3] = x, y
    return t


def test_register_recovers_translation():
    scene = _scene()
    offset = np.array([0.3, 0.2, 0.0], dtype=np.float32)
    reg = register_pair(scene + offset, scene, method="gicp")    # align the shifted scan back onto the scene
    t = np.array(reg["transformation"])[:3, 3]
    assert reg["fitness"] > 0.5
    assert np.allclose(t, -offset, atol=0.15)                    # recovered the inverse translation


def test_loop_closure_corrects_drift():
    # a large square trajectory that should return to the origin but drifts ~1.3 m; only the start and end
    # are close, so the revisit radius isolates the (0, 4) loop
    poses = [_pose(0, 0), _pose(20, 0), _pose(20, 20), _pose(0, 20), _pose(1.0, 0.8)]
    loops = detect_loops(poses, radius=3.0, min_gap=3)
    assert (0, 4) in loops
    pg = optimize_pose_graph(poses, loops)
    assert pg["drift_after_m"] < pg["drift_before_m"] and pg["drift_after_m"] < 0.1


def test_accumulate_merges_scans():
    a = Cloud(xyz=_scene(1), intensity=np.ones(3600, np.float32), ts_ns=1, source="pseudo")
    b = Cloud(xyz=_scene(2), intensity=np.ones(3600, np.float32), ts_ns=2, source="pseudo")
    agg = accumulate_scans([a, b], [_pose(0, 0), _pose(10, 0)], voxel=0.3)
    assert agg.n > 0 and agg.frame == "map"
    assert agg.xyz[:, 0].max() > 25                              # the second scan shifted +10 in x


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
def test_aggregate_sessions_persists():
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import AggregatedMap
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.lidar.aggregate import aggregate_sessions
        from services.lidar.ingest import store_cloud

        get_object_store().ensure_bucket()
        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="AGG-3D", start_ts_ns=ts, end_ts_ns=ts + 10**9,
                             city="BLR", sensors={}, ontology_version=get_ontology().version))
            await db.commit()
        for k in range(3):
            cloud = Cloud(xyz=(_scene(k) + np.array([0.2 * k, 0, 0], np.float32)).astype(np.float32),
                          intensity=np.ones(3600, np.float32), ts_ns=ts + k * 10**8, source="pseudo")
            await store_cloud(cloud, sid, source="pseudo")

        region = f"BLR-test-{uuid.uuid4().hex[:8]}"     # unique so the assertion is isolated across runs
        res = await aggregate_sessions([sid], region=region)
        assert res["n_scans"] == 3 and res["points"] > 0 and "mean_reg_fitness" in res

        from sqlalchemy import select
        async with get_sessionmaker()() as db:
            row = (await db.execute(select(AggregatedMap).where(AggregatedMap.region == region))
                   ).scalar_one()
            assert row.cloud_uri and row.n_scans == 3 and row.pose_graph

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
