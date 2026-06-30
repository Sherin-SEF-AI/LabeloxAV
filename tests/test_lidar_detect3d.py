"""M-L2.0: 2D objects lift into ground-snapped oriented cuboids with the right dimensions and yaw, every 3D
proposal passes the SAME governed gate (rare/fallback forces review, unsupported never invented), the native
path parks on the burst seam when OpenPCDet is absent, and a lifted cuboid persists linked to its 2D object.

Geometry, gate, and seam need no infra; the lift+persist test needs DB + MinIO."""

from __future__ import annotations

import asyncio
import math
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.lidar.detect3d import (
    NativeDetectionUnavailable,
    detect_native,
    fit_cuboid,
    frustum_indices,
    gate_cuboid,
    native_available,
    native_class_to_ontology,
)

W, H = 1280, 960


def _box_points(center, dims, yaw, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    local = rng.uniform(-0.5, 0.5, (n, 3)) * np.array(dims)
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (local @ r.T + np.array(center)).astype(np.float32)


def _yaw_aligned(fitted, truth):
    d = abs((fitted - truth + math.pi / 2) % math.pi - math.pi / 2)
    return d


def test_fit_cuboid_recovers_dims_and_ground_snaps():
    pts = _box_points([10.0, 2.0, 0.75], [4.0, 1.8, 1.5], yaw=0.0)
    cub = fit_cuboid(pts, ground_plane=[0.0, 0.0, 1.0, 0.0])
    assert cub is not None
    L, Wd, Ht = cub["dims"]
    assert 3.6 <= L <= 4.4 and 1.5 <= Wd <= 2.1 and 1.3 <= Ht <= 1.7
    cx, cy, cz = cub["center"]
    assert abs(cx - 10) < 0.4 and abs(cy - 2) < 0.4
    assert abs(cz - 0.75) < 0.2                       # bottom snapped to z=0, so centre at H/2
    assert _yaw_aligned(cub["yaw"], 0.0) < math.radians(12)


def test_fit_cuboid_recovers_orientation():
    pts = _box_points([14.0, -3.0, 0.8], [4.5, 1.9, 1.5], yaw=math.radians(35))
    cub = fit_cuboid(pts, ground_plane=[0.0, 0.0, 1.0, 0.0])
    assert _yaw_aligned(cub["yaw"], math.radians(35)) < math.radians(12)


def test_fit_cuboid_rejects_sparse():
    assert fit_cuboid(np.zeros((4, 3), np.float32), ground_plane=[0, 0, 1, 0]) is None


def test_frustum_selects_points_in_box():
    ahead = _box_points([12.0, 0.0, 0.6], [1.0, 1.0, 1.0], 0.0, n=500)   # straight ahead, projects to centre
    side = _box_points([6.0, 12.0, 0.6], [1.0, 1.0, 1.0], 0.0, n=500)    # far to the left, off-frame for cam_f
    cloud = np.vstack([ahead, side])
    box = [W / 2 - 120, H / 2 - 120, W / 2 + 120, H / 2 + 120]
    idx = frustum_indices(cloud, box, "cam_f", W, H)
    assert (idx < 500).mean() > 0.9                  # almost all selected points are the ahead cluster


def test_gate_carries_ontology_and_fallback_discipline():
    onto = get_ontology()
    car = onto.by_name("sedan").id
    fallback = onto.fallback_ids()[0]
    cub = {"center": [10, 0, 0.8], "dims": [4, 1.8, 1.5], "yaw": 0.0, "pitch": 0, "roll": 0,
           "fill": 0.9, "n_points": 2000}
    # a confident, agreed common class auto-accepts
    g = gate_cuboid(cub, class_id=car, conf_2d=0.98, frame_id=uuid.uuid4(), box_source="lifted",
                    agreement_2d=True)
    assert g["state"] == "auto_accept" and g["class_id"] == car
    # a fallback (rare) class never auto-accepts, whatever the score
    gr = gate_cuboid(cub, class_id=fallback, conf_2d=0.99, frame_id=uuid.uuid4(), box_source="lifted",
                     agreement_2d=True)
    assert gr["state"] == "review" and gr["is_fallback"] and gr["is_rare"]
    # below the review floor is a full annotate
    gl = gate_cuboid(cub, class_id=car, conf_2d=0.30, frame_id=uuid.uuid4(), box_source="lifted",
                     agreement_2d=True)
    assert gl["state"] == "annotate"


def test_native_class_maps_unsupported_to_fallback():
    onto = get_ontology()
    assert native_class_to_ontology("car") == onto.by_name("sedan").id
    fb = native_class_to_ontology("some_unknown_native_class")   # never invented
    assert onto.is_fallback(fb)


def test_native_detection_parks_on_seam_when_unavailable():
    assert native_available() is False                # OpenPCDet not installed on the interactive box
    from services.lidar.ingest import Cloud
    cloud = Cloud(xyz=np.zeros((10, 3), np.float32), intensity=np.zeros(10, np.float32), ts_ns=1, source="lidar")
    with pytest.raises(NativeDetectionUnavailable):
        detect_native(cloud)


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
def test_lift_frame_persists_object3d_linked_to_2d():
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Frame, Object, Object3D
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.detect3d import lift_frame
        from services.lidar.ingest import Cloud, store_cloud

        get_object_store().ensure_bucket()
        onto = get_ontology()
        car = onto.by_name("sedan").id
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        # a cloud with a car-shaped cluster ahead of cam_f, plus ground
        rng = np.random.default_rng(0)
        ground = np.stack([rng.uniform(2, 30, 4000), rng.uniform(-10, 10, 4000),
                           rng.normal(0, 0.02, 4000)], axis=1)
        car_pts = _box_points([12.0, 0.0, 0.75], [4.0, 1.8, 1.5], yaw=0.0, n=3000)
        cloud = Cloud(xyz=np.vstack([ground, car_pts]).astype(np.float32),
                      intensity=np.ones(7000, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="TIGOR-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/y.jpg",
                         width=W, height=H, quality=1.0))
            # a 2D box around where the car projects (centre of frame), high conf, agreed
            db.add(Object(frame_id=fid, class_id=car, bbox=[W / 2 - 200, H / 2 - 120, W / 2 + 200, H / 2 + 160],
                          conf=0.97, source="auto_accept", state="auto_accept",
                          provenance={"agreement": True}))
            await db.commit()
        await store_cloud(cloud, sid, source="pseudo")

        res = await lift_frame(fid)
        assert res["cuboids"] == 1
        o3d = res["objects"][0]
        assert o3d["object_id"] is not None and o3d["class_id"] == car      # linked to the 2D object identity
        assert 3.5 <= o3d["dims"][0] <= 4.5                                  # recovered the car length

        # idempotent: re-running does not duplicate machine 3D objects
        await lift_frame(fid)
        from sqlalchemy import func, select
        async with get_sessionmaker()() as db:
            n = (await db.execute(select(func.count()).select_from(Object3D)
                 .where(Object3D.frame_id == fid))).scalar()
            assert n == 1

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()


@requires_infra
def test_lift_is_per_frame_not_per_cloud_multicamera():
    """Two cameras share one fused cloud at a ts_ns. Re-lifting one camera must not wipe the other's cuboids
    (the idempotent clear is scoped to the frame, not the shared cloud)."""
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import Frame, Object, Object3D
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.lidar.detect3d import lift_frame
        from services.lidar.ingest import Cloud, store_cloud

        get_object_store().ensure_bucket()
        onto = get_ontology()
        car = onto.by_name("sedan").id
        sid, ts = uuid.uuid4(), now_ns()
        ff, fb = uuid.uuid4(), uuid.uuid4()
        rng = np.random.default_rng(0)
        ground = np.stack([rng.uniform(2, 30, 4000), rng.uniform(-10, 10, 4000),
                           rng.normal(0, 0.02, 4000)], axis=1)
        car_pts = _box_points([12.0, 0.0, 0.75], [4.0, 1.8, 1.5], yaw=0.0, n=3000)
        cloud = Cloud(xyz=np.vstack([ground, car_pts]).astype(np.float32),
                      intensity=np.ones(7000, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="MC-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            db.add(Frame(frame_id=ff, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x.jpg",
                         width=W, height=H, quality=1.0))
            db.add(Frame(frame_id=fb, session_id=sid, ts_ns=ts, cam_id="cam_b", img_uri="s3://x.jpg",
                         width=W, height=H, quality=1.0))
            db.add(Object(frame_id=ff, class_id=car, bbox=[W / 2 - 200, H / 2 - 120, W / 2 + 200, H / 2 + 160],
                          conf=0.97, source="auto_accept", state="auto_accept", provenance={"agreement": True}))
            await db.commit()
        stored = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(stored["cloud_id"])

        await lift_frame(ff)
        # cam_b lifted its own machine cuboid earlier on the SAME shared cloud (inserted directly)
        async with get_sessionmaker()() as db:
            db.add(Object3D(cloud_id=cloud_id, frame_id=fb, class_id=car, center=[-12, 0, 0.75],
                            dims=[4, 1.8, 1.5], yaw=0.0, conf=0.9, box_source="lifted", source="fused",
                            state="auto_accept"))
            await db.commit()

        await lift_frame(ff)                       # re-lift the front camera on the shared cloud
        from sqlalchemy import func, select
        async with get_sessionmaker()() as db:
            n_f = (await db.execute(select(func.count()).select_from(Object3D)
                   .where(Object3D.frame_id == ff))).scalar()
            n_b = (await db.execute(select(func.count()).select_from(Object3D)
                   .where(Object3D.frame_id == fb))).scalar()
            assert n_f == 1 and n_b == 1          # the rear camera's lift was not wiped by the front re-lift

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
