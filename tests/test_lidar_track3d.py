"""M-L2.2: the 3D tracker keeps one identity across frames and links to the 2D track, keyframe interpolation
fills the pose between keyframes, and ego-compensated dynamics tells a parked object from a moving one.

Tracker, interpolation, and dynamics need no infra; the session orchestration needs the DB."""

from __future__ import annotations

import asyncio
import math
import uuid

import pytest

from core.config import get_settings
from services.lidar.track3d import (
    Tracker3D,
    classify_track,
    interpolate_cuboids,
)

DIMS = [4.0, 1.8, 1.5]


def test_tracker_keeps_identity_and_links_2d_track():
    tr = Tracker3D(iou_thresh=0.1, min_hits=1)
    ids = []
    for k in range(5):
        det = {"center": [10 + k * 0.5, 0.0, 0.75], "dims": DIMS, "yaw": 0.0, "class_id": 11,
               "track_id_2d": "track-abc", "object_3d_id": uuid.uuid4()}
        out = tr.step([det], dt=0.1)
        ids.append(out[0]["track_3d_local_id"])
    assert len(set(ids)) == 1                          # one consistent 3D identity across frames
    confirmed = tr.confirmed()
    assert len(confirmed) == 1 and confirmed[0].linked_track_2d() == "track-abc"


def test_tracker_separates_distinct_objects():
    tr = Tracker3D(iou_thresh=0.1, min_hits=1)
    out = tr.step([
        {"center": [10, 0, 0.75], "dims": DIMS, "yaw": 0.0, "class_id": 11, "object_3d_id": uuid.uuid4()},
        {"center": [10, 20, 0.75], "dims": DIMS, "yaw": 0.0, "class_id": 11, "object_3d_id": uuid.uuid4()},
    ], dt=0.1)
    assert len({o["track_3d_local_id"] for o in out}) == 2


def test_keyframe_interpolation_fills_pose():
    kf = [{"ts_ns": 0, "center": [0, 0, 0], "dims": [4, 2, 1.5], "yaw": 0.0},
          {"ts_ns": 100, "center": [10, 0, 0], "dims": [4, 2, 1.5], "yaw": math.pi / 2}]
    out = interpolate_cuboids(kf, [0, 50, 100])        # the two keyframes are skipped, the middle is filled
    assert len(out) == 1 and out[0]["ts_ns"] == 50
    assert abs(out[0]["center"][0] - 5.0) < 1e-6
    assert abs(out[0]["yaw"] - math.pi / 4) < 1e-3 and out[0]["interp_source"] == "linear"


def test_dynamics_parked_vs_moving_with_ego_compensation():
    # a parked object: the vehicle drives forward at 10 m/s, so the static object slides back 1 m per frame
    parked = [{"ts_ns": k * 100_000_000, "center": [20 - k * 1.0, 3, 0.75], "yaw": 0.0, "ego_speed": 10.0}
              for k in range(5)]
    assert classify_track(parked)["state"] == "parked"
    # a moving object closing 5 m/s slower than the ego: ego-frame recession is only 0.5 m per frame
    moving = [{"ts_ns": k * 100_000_000, "center": [20 - k * 0.5, 3, 0.75], "yaw": 0.0, "ego_speed": 10.0}
              for k in range(5)]
    assert classify_track(moving)["state"] == "moving"


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
def test_track_session_links_and_classifies():
    async def run():
        from db.models import Frame, Object, Object3D, PointCloud, Track, Track3D
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.lidar.track3d import track_session

        onto = get_ontology()
        sedan = onto.by_name("sedan").id
        sid, tid2d = uuid.uuid4(), uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="TRK-3D", start_ts_ns=0, end_ts_ns=10**9,
                             city="BLR", sensors={}, ontology_version=onto.version))
            db.add(Track(track_id=tid2d, session_id=sid, class_id=sedan, first_ts_ns=0, last_ts_ns=4 * 10**8))
            await db.flush()
            for k in range(5):
                ts = k * 10**8
                fid, cid = uuid.uuid4(), uuid.uuid4()
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x.jpg",
                             width=1280, height=960, quality=1.0, ego_speed=10.0))
                db.add(PointCloud(cloud_id=cid, session_id=sid, ts_ns=ts, source="pseudo",
                                  cloud_uri=f"s3://c/{cid}.npz", point_count=100, bounds={}))
                oid2d = uuid.uuid4()
                db.add(Object(object_id=oid2d, frame_id=fid, track_id=tid2d, class_id=sedan,
                              bbox=[0, 0, 10, 10], conf=0.9, source="fused", state="auto_accept"))
                await db.flush()                                 # parents exist before the object_3d FK
                # a parked car: static in the world, sliding back 1 m per 0.1 s as the ego advances
                db.add(Object3D(cloud_id=cid, frame_id=fid, object_id=oid2d, class_id=sedan,
                                center=[20 - k * 1.0, 3.0, 0.75], dims=DIMS, yaw=0.0, conf=0.9,
                                box_source="lifted", source="fused", state="auto_accept"))
            await db.commit()

        res = await track_session(sid)
        assert res["tracks"] == 1

        from sqlalchemy import select
        async with get_sessionmaker()() as db:
            tr = (await db.execute(select(Track3D).where(Track3D.session_id == sid))).scalars().all()
            assert len(tr) == 1
            assert str(tr[0].track_id) == str(tid2d)        # linked to the 2D track
            assert tr[0].dynamic_state == "parked"          # ego-compensated: static in the world
            n_linked = (await db.execute(select(Object3D).where(Object3D.track_3d_id == tr[0].track_3d_id))
                        ).scalars().all()
            assert len(n_linked) == 5                        # every cuboid carries the 3D track id

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
