"""M-L2.1: cuboid geometry (corners, ground snap, camera projection, 3D IoU) and the cuboid CRUD workspace
(create ground-snapped with an ontology class, edit with optimistic locking, project onto the camera, delete).

Geometry needs no infra; the CRUD endpoints need DB + MinIO."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.boxes import cuboid_corners, iou_3d, project_cuboid, snap_to_ground

W, H = 1280, 960


def test_cuboid_corners_axis_aligned():
    corners = cuboid_corners([0, 0, 0], [4, 2, 1.5], yaw=0.0)
    assert corners.shape == (8, 3)
    assert set(np.round(corners[:, 0], 3)) == {-2.0, 2.0}
    assert set(np.round(corners[:, 1], 3)) == {-1.0, 1.0}
    assert set(np.round(corners[:, 2], 3)) == {-0.75, 0.75}
    assert np.allclose(corners.mean(axis=0), [0, 0, 0], atol=1e-5)


def test_snap_to_ground():
    c = snap_to_ground([10.0, 2.0, 5.0], [4.0, 2.0, 1.5], [0.0, 0.0, 1.0, 0.0])
    assert c == [10.0, 2.0, 0.75]                      # bottom on z=0, centre at H/2


def test_project_cuboid_onto_camera():
    proj = project_cuboid([12.0, 0.0, 0.75], [4.0, 2.0, 1.5], 0.0, "cam_f", W, H)
    assert proj["any_in_image"] and len(proj["corners_uv"]) == 8 and len(proj["edges"]) == 12
    uv = np.array(proj["corners_uv"])
    assert (abs(uv[:, 0] - W / 2) < W / 2).all()       # corners land within the frame width


def test_iou_3d():
    a = {"center": [0, 0, 0], "dims": [4, 2, 2], "yaw": 0.0}
    assert abs(iou_3d(a, a) - 1.0) < 1e-3              # identical boxes
    far = {"center": [10, 0, 0], "dims": [4, 2, 2], "yaw": 0.0}
    assert iou_3d(a, far) == 0.0                       # disjoint
    half = {"center": [2, 0, 0], "dims": [4, 2, 2], "yaw": 0.0}
    assert 0.2 < iou_3d(a, half) < 0.5                 # partial overlap


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


async def _seed_cloud() -> tuple[uuid.UUID, int]:
    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.lidar.ingest import Cloud, store_cloud

    get_object_store().ensure_bucket()
    sid, ts = uuid.uuid4(), now_ns()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="ANNO-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/y.jpg",
                     width=W, height=H, quality=1.0))
        await db.commit()
    rng = np.random.default_rng(0)
    ground = np.stack([rng.uniform(2, 30, 3000), rng.uniform(-10, 10, 3000), rng.normal(0, 0.02, 3000)], axis=1)
    cloud = Cloud(xyz=ground.astype(np.float32), intensity=np.ones(3000, np.float32), ts_ns=ts, source="pseudo")
    res = await store_cloud(cloud, sid, source="pseudo")
    from db.session import get_engine
    await get_engine().dispose()                   # close pooled connections on this loop before it ends
    return uuid.UUID(res["cloud_id"]), ts


@requires_infra
def test_cuboid_crud_ground_snap_and_locking():
    from fastapi.testclient import TestClient

    from services.api.main import app
    from services.autolabel.ontology import get_ontology

    _clear()
    cloud_id, _ = asyncio.run(_seed_cloud())
    _clear()                                       # rebuild the engine for the TestClient's event loop
    sedan = get_ontology().by_name("sedan").id
    try:
        # the context-manager form holds one event loop across every request (else the async engine binds
        # to a per-request loop that is closed by the next request)
        with TestClient(app) as c:
            created = c.post(f"/api/lidar/clouds/{cloud_id}/objects3d", json={
                "class_id": sedan, "center": [12.0, 0.0, 9.0], "dims": [4.0, 1.8, 1.5], "yaw": 0.3,
                "ground_snap": True}).json()
            oid = created["object_3d_id"]
            assert created["source"] == "human" and created["class_name"] == "sedan" and created["is_keyframe"]
            assert abs(created["center"][2] - 0.75) < 0.25      # snapped to ground (H/2 above z=0)
            assert created["version"] == 1

            listed = c.get(f"/api/lidar/clouds/{cloud_id}/objects3d").json()
            assert len(listed["objects"]) == 1

            # edit with the right version bumps it; a stale version is a 409
            edited = c.patch(f"/api/lidar/objects3d/{oid}", json={"yaw": 1.2, "expected_version": 1}).json()
            assert edited["version"] == 2 and abs(edited["yaw"] - 1.2) < 1e-6
            stale = c.patch(f"/api/lidar/objects3d/{oid}", json={"yaw": 0.0, "expected_version": 1})
            assert stale.status_code == 409

            proj = c.get(f"/api/lidar/objects3d/{oid}/projection",
                         params={"cam_id": "cam_f", "w": W, "h": H}).json()
            assert len(proj["corners_uv"]) == 8 and proj["any_in_image"]

            assert c.delete(f"/api/lidar/objects3d/{oid}").status_code == 200
            assert len(c.get(f"/api/lidar/clouds/{cloud_id}/objects3d").json()["objects"]) == 0
    finally:
        _clear()
