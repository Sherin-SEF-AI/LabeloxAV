"""M-L1.2: ground removal isolates the road surface, denoising drops outliers and rain/dust, and the full
pipeline writes derived variants without touching the raw cloud.

Geometry tests need open3d only. The pipeline test needs the DB + MinIO (requires_infra)."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.clean import (
    denoise,
    filter_rain_dust,
    remove_ground,
    segment_ground,
    statistical_outlier,
)
from services.lidar.ingest import Cloud


def _o3d_ok() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except Exception:
        return False


requires_o3d = pytest.mark.skipif(not _o3d_ok(), reason="open3d unavailable")


def _ground_plus_objects(seed: int = 0) -> Cloud:
    rng = np.random.default_rng(seed)
    gx = rng.uniform(-20, 20, 3000)
    gy = rng.uniform(-20, 20, 3000)
    gz = rng.normal(0.0, 0.02, 3000)                      # the road plane near z=0
    ground = np.stack([gx, gy, gz], axis=1)
    ox = rng.uniform(-2, 2, 600)
    oy = rng.uniform(3, 7, 600)
    oz = rng.uniform(0.6, 2.5, 600)                       # objects standing above the road
    objs = np.stack([ox, oy, oz], axis=1)
    xyz = np.vstack([ground, objs]).astype(np.float32)
    return Cloud(xyz=xyz, intensity=rng.uniform(0.2, 1.0, xyz.shape[0]).astype(np.float32), ts_ns=1)


@requires_o3d
def test_ground_segmentation_isolates_road():
    cloud = _ground_plus_objects()
    mask, plane, used = segment_ground(cloud, method="ransac", dist_thresh=0.15)
    assert used == "ransac"
    assert abs(plane[2]) / (np.linalg.norm(plane[:3]) or 1) > 0.9   # near-horizontal plane
    assert mask.sum() > 2500                                        # most of the 3000 ground points found
    # the standing objects survive the ground removal
    res = remove_ground(cloud, dist_thresh=0.15)
    assert res["kept_points"] >= 550 and res["ground_points"] > 2500
    assert (res["nonground"].xyz[:, 2] > 0.4).mean() > 0.8         # kept points are mostly the objects


@requires_o3d
def test_statistical_outlier_removes_speckle():
    rng = np.random.default_rng(1)
    dense = rng.normal(0, 0.5, (2000, 3))
    speckle = rng.uniform(-50, 50, (50, 3))                        # far isolated noise
    cloud = Cloud(xyz=np.vstack([dense, speckle]).astype(np.float32),
                  intensity=np.ones(2050, np.float32), ts_ns=1)
    cleaned = statistical_outlier(cloud, nb_neighbors=20, std_ratio=2.0)
    assert cloud.n - cleaned.n >= 30                                # most speckle dropped
    assert cleaned.n >= 1950                                        # the dense core is kept


@requires_o3d
def test_rain_dust_keeps_bright_or_dense():
    rng = np.random.default_rng(2)
    dense = rng.normal(0, 0.3, (1500, 3)).astype(np.float32)       # well supported
    sparse = rng.uniform(-30, 30, (200, 3)).astype(np.float32)     # isolated
    xyz = np.vstack([dense, sparse])
    inten = np.concatenate([rng.uniform(0.65, 1.0, 1500),          # dense points are bright
                            rng.uniform(0.0, 0.1, 200)]).astype(np.float32)  # sparse points are dim
    cloud = Cloud(xyz=xyz, intensity=inten, ts_ns=1)
    out = filter_rain_dust(cloud, intensity_pct=25.0, radius=0.5, min_neighbors=4)
    assert out.n < cloud.n                                          # dim+isolated points removed
    assert out.n >= 1500                                           # bright-or-dense core kept


@requires_o3d
def test_denoise_runs_full_pass():
    out = denoise(_ground_plus_objects(), rain_dust=True)
    assert 0 < out.n <= 3600


def _infra_up() -> bool:
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not (_infra_up() and _o3d_ok()), reason="infra/open3d not up")


def _clear():
    from db.session import get_engine, get_sessionmaker
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@requires_infra
def test_clean_cloud_writes_derived_leaves_raw_untouched():
    async def run():
        from core.storage import get_object_store
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.clean import clean_cloud
        from services.lidar.ingest import store_cloud

        get_object_store().ensure_bucket()
        sid = uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="LIDAR-CLEAN", start_ts_ns=0, end_ts_ns=1,
                             city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
            await db.commit()
        raw = _ground_plus_objects()
        stored = await store_cloud(raw, sid, source="dataset")
        cloud_id = uuid.UUID(stored["cloud_id"])

        res = await clean_cloud(cloud_id, sid, method="ransac")
        assert set(res["derived"]) == {"ground_plane", "ground_removed", "denoised"}
        assert res["derived"]["ground_removed"]["points"] > 400

        # raw point_cloud row is unchanged and three derived rows exist
        from sqlalchemy import func, select

        from db.models import PointCloud, PointCloudDerived
        async with get_sessionmaker()() as db:
            pc = await db.get(PointCloud, cloud_id)
            assert pc.point_count == raw.n
            n_derived = (await db.execute(
                select(func.count()).select_from(PointCloudDerived)
                .where(PointCloudDerived.cloud_id == cloud_id))).scalar()
            assert n_derived == 3

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
