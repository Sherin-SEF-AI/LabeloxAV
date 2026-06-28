"""M-L1.0: every point cloud source normalizes to one internal Cloud, round-trips through the object store
and MCAP, and a stored cloud is queryable alongside the camera frames captured at the same PPS ts_ns.

The pure-unit tests need no infra. The store test needs the DB + MinIO (requires_infra)."""

from __future__ import annotations

import asyncio
import io
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.ingest import (
    Cloud,
    read_kitti_bin,
    read_las,
    read_nuscenes_bin,
    read_pcd,
    read_pointclouds_mcap,
    write_pointclouds_mcap,
)


def _sample_xyzi(n: int = 50, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-20, 60, size=(n, 3)).astype(np.float32)
    inten = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    return np.hstack([xyz, inten])


def test_all_sources_one_representation():
    pts = _sample_xyzi()
    xyz = pts[:, :3]

    # KITTI .bin bytes -> Cloud
    kitti = read_kitti_bin(pts.astype(np.float32).tobytes())
    assert kitti.xyz.shape == (50, 3) and kitti.source == "dataset" and kitti.frame == "kitti_velo"
    assert np.allclose(kitti.xyz, xyz, atol=1e-4)

    # nuScenes .bin (x,y,z,intensity,ring) -> Cloud with a ring
    nus = np.hstack([pts, np.arange(50).reshape(-1, 1).astype(np.float32)])
    nu = read_nuscenes_bin(nus.astype(np.float32).tobytes())
    assert nu.ring is not None and nu.ring.shape == (50,) and np.allclose(nu.xyz, xyz, atol=1e-4)

    # PCD -> Cloud, same xyz
    from pypcd4 import PointCloud as PCD
    buf = io.BytesIO()
    PCD.from_xyzi_points(pts.astype(np.float32)).save(buf)
    pcd = read_pcd(buf.getvalue())
    assert pcd.xyz.shape == (50, 3) and np.allclose(pcd.xyz, xyz, atol=1e-3)

    # LAS -> Cloud, same xyz
    import laspy
    las = laspy.LasData(laspy.LasHeader(point_format=3))
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lb = io.BytesIO()
    las.write(lb)
    lasc = read_las(lb.getvalue())
    assert lasc.xyz.shape == (50, 3) and np.allclose(lasc.xyz, xyz, atol=1e-2)


def test_cloud_npz_roundtrip():
    pts = _sample_xyzi()
    c = Cloud(xyz=pts[:, :3], intensity=pts[:, 3], ts_ns=1234567890, ring=np.arange(50).astype(np.int16),
              source="pseudo", frame="ego", depth_model="depth-anything-v2", calibration_version="calib-1")
    back = Cloud.from_npz_bytes(c.to_npz_bytes())
    assert back.ts_ns == 1234567890 and back.source == "pseudo" and back.depth_model == "depth-anything-v2"
    assert np.allclose(back.xyz, c.xyz) and np.array_equal(back.ring, c.ring)
    b = c.bounds()
    assert b["n"] == 50 and len(b["min"]) == 3 and len(b["max"]) == 3


def test_mcap_roundtrip(tmp_path):
    a = Cloud(xyz=_sample_xyzi(30, 1)[:, :3], intensity=np.zeros(30, np.float32), ts_ns=2000, source="lidar")
    b = Cloud(xyz=_sample_xyzi(40, 2)[:, :3], intensity=np.zeros(40, np.float32), ts_ns=1000, source="lidar")
    res = write_pointclouds_mcap([a, b], tmp_path / "clouds.mcap")
    assert res["clouds"] == 2 and res["topic"] == "/points"
    got = list(read_pointclouds_mcap(tmp_path / "clouds.mcap"))
    assert [g.ts_ns for g in got] == [1000, 2000]  # ordered by ts_ns
    assert got[0].n == 40 and got[1].n == 30 and np.allclose(got[1].xyz, a.xyz, atol=1e-4)


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


@requires_infra
def test_store_cloud_links_frames_by_ts():
    """A stored cloud and the camera frames at the same ts_ns are one query."""
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.ingest import load_cloud, store_cloud

        store = get_object_store()
        store.ensure_bucket()
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="LIDAR-TEST", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                         img_uri="s3://x/y.jpg", width=640, height=480, quality=1.0))
            await db.commit()
        cloud = Cloud(xyz=_sample_xyzi(100)[:, :3], intensity=np.zeros(100, np.float32), ts_ns=ts,
                      source="pseudo", depth_model="depth-anything-v2", calibration_version="calib-1")
        res = await store_cloud(cloud, sid)
        assert res["point_count"] == 100
        assert any(f["frame_id"] == str(fid) and f["cam_id"] == "cam_f" for f in res["synced_frames"])
        # the stored cloud reads back identically
        back = load_cloud(res["cloud_uri"])
        assert back.n == 100 and back.ts_ns == ts and np.allclose(back.xyz, cloud.xyz, atol=1e-4)

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
