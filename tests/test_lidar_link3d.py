"""M-L2.4: a 3D cuboid projects to a 2D box and links to the 2D object (one identity across cloud and every
camera), each 3D object carries auto-computed distance, heading, velocity, and occlusion, and correcting one
object finds similar ones and batch-updates them.

Projection and properties need no infra; linking and correction need the DB."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.lidar.link import compute_object_properties, projected_bbox
from services.lidar.link.object_identity import _iou

W, H = 1280, 960


def test_projected_bbox_and_iou():
    bbox = projected_bbox([12.0, 0.0, 0.75], [4.0, 1.8, 1.5], 0.0, "cam_f", W, H)
    assert bbox is not None and bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert _iou(bbox, bbox) == 1.0
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_auto_properties_distance_and_ego_compensated_velocity():
    # a parked object: it slides forward->back in the ego frame as the vehicle advances at 10 m/s
    traj = [{"ts_ns": 0, "center": [13.0, 0, 0.75]}, {"ts_ns": 100_000_000, "center": [12.0, 0, 0.75]}]
    props = compute_object_properties([12.0, 0.0, 0.75], [4.0, 1.8, 1.5], 0.0, trajectory=traj,
                                      ego_speed=10.0, in_image_frac=1.0, points_in_box=300)
    assert props["distance_m"] == 12.0 and props["heading_deg"] == 0.0
    assert abs(props["velocity_mps"]) < 0.6                 # ego-compensated: parked reads ~0
    assert props["occlusion"] == 0.0                        # fully in image, dense

    occluded = compute_object_properties([40.0, 0.0, 0.75], [4.0, 1.8, 1.5], 0.0,
                                         in_image_frac=0.3, points_in_box=2)
    assert occluded["occlusion"] > 0.5                      # mostly out of frame and sparse


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
def test_link_identity_and_batch_correct():
    async def run():
        from core.timebase import now_ns
        from db.models import Frame, Object, Object3D, PointCloud
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.lidar.link import batch_correct, find_similar, link_cloud, linked_views

        onto = get_ontology()
        sedan, suv = onto.by_name("sedan").id, onto.by_name("suv").id
        sid, ts = uuid.uuid4(), now_ns()
        cuboid = ([12.0, 0.0, 0.75], [4.0, 1.8, 1.5], 0.0)
        bbox2d = projected_bbox(*cuboid, "cam_f", W, H)
        cloud_id, oid2d, native_o3d = uuid.uuid4(), uuid.uuid4(), None
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="LINK-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.flush()                                 # session exists before its children
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x.jpg",
                         width=W, height=H, quality=1.0, ego_speed=8.0))
            db.add(PointCloud(cloud_id=cloud_id, session_id=sid, ts_ns=ts, source="pseudo",
                              cloud_uri=f"s3://c/{cloud_id}.npz", point_count=100, bounds={}))
            db.add(Object(object_id=oid2d, frame_id=fid, class_id=sedan, bbox=bbox2d, conf=0.9,
                          source="fused", state="auto_accept"))
            await db.flush()
            # a native cuboid with no 2D link yet, projecting onto the 2D object
            o = Object3D(cloud_id=cloud_id, frame_id=fid, class_id=sedan, center=cuboid[0], dims=cuboid[1],
                         yaw=cuboid[2], conf=0.9, box_source="native", source="fused", state="review")
            db.add(o)
            await db.flush()
            native_o3d = o.object_3d_id
            await db.commit()

        res = await link_cloud(cloud_id)
        assert res["linked"] == 1
        lv = await linked_views(native_o3d)
        assert lv["object_id"] == str(oid2d) and "cam_f" in lv["projections"] and lv["object_2d"] is not None

        sim = await find_similar(native_o3d, k=5)
        assert sim["class_id"] == sedan
        upd = await batch_correct([native_o3d], class_id=suv)
        assert upd["updated"] == 1
        async with get_sessionmaker()() as db:
            o = await db.get(Object3D, native_o3d)
            assert o.class_id == suv and o.source == "human"

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
