"""M-L1.5: calibration validation reports reprojection and consistency residuals and flags drift, the
quality check flags a sparse, partial, or dead-channel cloud, and a failing session is excluded from 3D work.

Residual and quality maths need no infra; the session validation, drift, and exclusion gate need DB + MinIO."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.calib.lidar_camera import reprojection_error
from services.lidar.clean.qualitypc import check_cloud_quality
from services.lidar.ingest import Cloud
from services.lidar.project import project_to_camera

W, H = 1280, 960


def test_reprojection_error_detects_mismatch():
    pts = np.array([[10, 1, 0.5], [15, -2, 1.0], [8, 0, 0.8]], dtype=np.float32)
    uv = project_to_camera(pts, "cam_f", W, H)["uv"]
    good = reprojection_error(pts, uv, "cam_f", W, H)
    assert good["rms"] < 1e-3 and good["n"] == 3          # a matched calibration has ~zero residual
    shift = np.array([10.0, 0.0], dtype=np.float32)
    bad = reprojection_error(pts, uv + shift, "cam_f", W, H)
    assert 9.0 < bad["rms"] < 11.0                        # a 10 px offset shows as a 10 px residual


def _ring_cloud(source="lidar", full=True, n=5000, rings=None) -> Cloud:
    rng = np.random.default_rng(0)
    az = rng.uniform(-np.pi, np.pi, n) if full else rng.uniform(-np.pi / 3, np.pi / 3, n)
    r = rng.uniform(3, 40, n)
    xyz = np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-1, 3, n)], axis=1).astype(np.float32)
    ring = np.asarray(rings) if rings is not None else None
    return Cloud(xyz=xyz, intensity=rng.uniform(0, 1, n).astype(np.float32), ts_ns=1, ring=ring, source=source)


def test_quality_passes_full_scan():
    assert check_cloud_quality(_ring_cloud(full=True))["status"] == "pass"


def test_quality_flags_partial_scan():
    q = check_cloud_quality(_ring_cloud(full=False))     # only the front 120 deg of a spinning LiDAR
    assert q["checks"]["partial_scan"] and q["status"] == "fail"
    assert q["largest_empty_wedge_deg"] > 90


def test_quality_flags_sparse_and_missing():
    assert check_cloud_quality(_ring_cloud(n=100))["checks"]["sparse"]
    empty = Cloud(xyz=np.zeros((0, 3), np.float32), intensity=np.zeros(0, np.float32), ts_ns=1, source="lidar")
    assert check_cloud_quality(empty)["checks"]["missing_scan"]


def test_quality_flags_dead_channel():
    rings = np.array([0, 1, 2, 5, 6, 7] * 900, dtype=np.int16)[:5000]   # channels 3 and 4 are dead
    q = check_cloud_quality(_ring_cloud(rings=rings))
    assert q["dead_channels"] == 2 and q["checks"]["dead_channels"]


def test_pseudo_lidar_front_wedge_is_not_partial():
    # a forward-only camera cloud is not a defect; partial only applies to a 360 sensor
    q = check_cloud_quality(_ring_cloud(source="pseudo", full=False))
    assert not q["checks"]["partial_scan"] and q["status"] == "pass"


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
def test_session_validation_drift_and_exclusion():
    async def run():
        from core.storage import get_object_store
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.calib.validate3d import (
            lidar_session_ok,
            validate_lidar_camera,
            validate_session,
        )
        from services.lidar.ingest import store_cloud

        get_object_store().ensure_bucket()

        async def mk_session(vid: str) -> uuid.UUID:
            sid = uuid.uuid4()
            async with get_sessionmaker()() as db:
                db.add(DbSession(session_id=sid, vehicle_id=vid, start_ts_ns=0, end_ts_ns=1, city="BLR",
                                 sensors={}, ontology_version="labelox-in-0.1.0"))
                await db.commit()
            return sid

        # healthy session: a full 360 cloud validates pass and is not excluded
        good = await mk_session("LIDAR-GOOD")
        await store_cloud(_ring_cloud(full=True), good, source="lidar")
        v_good = await validate_session(good)
        assert v_good["status"] in ("pass", "warn")
        assert await lidar_session_ok(good) is True

        # broken session: a sparse cloud fails quality, so the session is excluded from 3D work
        bad = await mk_session("LIDAR-BAD")
        await store_cloud(_ring_cloud(n=80, full=True), bad, source="lidar")
        v_bad = await validate_session(bad)
        assert v_bad["status"] == "fail"
        assert await lidar_session_ok(bad) is False

        # drift: a residual that grows past the baseline ratio is flagged
        drift_sid = await mk_session("LIDAR-DRIFT")
        pts = np.array([[10, 1, 0.5], [15, -2, 1.0], [8, 0, 0.8]], dtype=np.float32)
        uv = project_to_camera(pts, "cam_f", W, H)["uv"]
        base = await validate_lidar_camera(drift_sid, pts, uv + np.array([1.0, 0.0], np.float32), "cam_f", W, H)
        assert base["status"] == "pass"                    # residual 1.0 px, within the warn threshold
        drifted = await validate_lidar_camera(drift_sid, pts, uv + np.array([3.0, 0.0], np.float32), "cam_f", W, H)
        assert drifted["drift_flag"] is True               # 3.0 px > 1.5x the 1.0 px baseline

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
