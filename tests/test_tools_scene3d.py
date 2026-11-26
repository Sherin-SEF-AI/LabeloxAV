"""Agent C's 3D tools: label once in 3D (#3), point labels (#5), occupancy editing (#6), plus the KITTI
calibration fix they stand on.

Pure geometry (KITTI extrinsics, scene-frame transforms, clipped cuboid projection, label merging, occupancy
delta algebra) runs without a database. The write paths run against the test database on generated scenes
whose answer is known: a static box seen from a moving vehicle, a ground plane with one object on it.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---- KITTI calibration -------------------------------------------------------------------------------------

# The real 2011_09_26 calibration (KITTI raw, CC BY-NC-SA 3.0), so the test does not depend on the dataset
# being on disk.
_C2C = """calib_time: 09-Jan-2012 13:57:47
R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
S_rect_02: 1.242000e+03 3.750000e+02
P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
"""
_V2C = """calib_time: 15-Mar-2012 11:37:16
R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02
T: -4.069766e-03 -7.631618e-02 -2.717806e-01
"""
_I2V = """calib_time: 25-May-2012 16:47:16
R: 9.999976e-01 7.553071e-04 -2.035826e-03 -7.854027e-04 9.998898e-01 -1.482298e-02 2.024406e-03 1.482454e-02 9.998881e-01
T: -8.086759e-01 3.195559e-01 -7.997231e-01
"""


def _calib_dir(tmp_path: Path) -> Path:
    (tmp_path / "calib_cam_to_cam.txt").write_text(_C2C)
    (tmp_path / "calib_velo_to_cam.txt").write_text(_V2C)
    (tmp_path / "calib_imu_to_velo.txt").write_text(_I2V)
    return tmp_path


def test_kitti_calibration_reproduces_kittis_own_projection(tmp_path):
    from services.calibration.resolve import Calibration
    from services.lidar.calib.kitti import kitti_camera_calibration, read_calib_file
    from services.lidar.project import project_to_camera

    d = _calib_dir(tmp_path)
    k = kitti_camera_calibration(d)
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(4, 60, 3000), rng.uniform(-15, 15, 3000), rng.uniform(-1.8, 2.0, 3000)], 1)
    c2c, v2c = read_calib_file(d / "calib_cam_to_cam.txt"), read_calib_file(d / "calib_velo_to_cam.txt")
    tr = np.eye(4)
    tr[:3, :3], tr[:3, 3] = v2c["R"].reshape(3, 3), v2c["T"]
    r0 = np.eye(4)
    r0[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    uvw = (c2c["P_rect_02"].reshape(3, 4) @ r0 @ tr @ np.c_[pts, np.ones(len(pts))].T).T
    ref = uvw[:, :2] / uvw[:, 2:3]
    cal = Calibration(cam_id="cam_front", model="pinhole", fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"],
                      rpy_deg=tuple(k["rpy_deg"]), xyz_m=tuple(k["xyz_m"]), source="dataset", quality=0.95)
    out = project_to_camera(pts.astype(np.float32), "cam_front", 1242, 375, cal)
    m = out["in_image"]
    assert m.sum() > 500
    assert np.abs(out["uv"][m] - ref[m]).max() < 0.01          # sub-centipixel against KITTI's formula
    # The camera sits 27 cm ahead of and 7 cm below the Velodyne, tilted ~0.6 degrees: not the old guess.
    assert k["xyz_m"][0] == pytest.approx(0.27, abs=0.01) and k["xyz_m"][2] == pytest.approx(-0.072, abs=0.01)
    assert abs(k["rpy_deg"][1]) < 2 and abs(k["rpy_deg"][0]) < 2


def test_extrinsics_round_trip_any_rotation():
    from services.calibration.resolve import Calibration
    from services.lidar.calib.kitti import extrinsics_from_affine

    rng = np.random.default_rng(1)
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        c = rng.normal(size=3)
        rpy, xyz = extrinsics_from_affine(q, c)
        cal = Calibration(cam_id="x", model="pinhole", fx=1, fy=1, cx=0, cy=0, rpy_deg=tuple(rpy), xyz_m=tuple(xyz))
        p = rng.normal(size=(10, 3))
        assert np.allclose((p - cal.t()) @ cal.R(), p @ q.T + c, atol=1e-4)


def test_imu_lever_arm_inverts(tmp_path):
    from services.lidar.calib.kitti import kitti_imu_to_velo, velo_pose_in_imu

    i2v = kitti_imu_to_velo(_calib_dir(tmp_path))
    t = velo_pose_in_imu(i2v)
    fwd = np.eye(4)
    fwd[:3, :3], fwd[:3, 3] = np.asarray(i2v["R"]), np.asarray(i2v["T"])
    assert np.allclose(fwd @ t, np.eye(4), atol=1e-9)
    assert np.allclose(velo_pose_in_imu(None), np.eye(4))


# ---- scene frame and label-once geometry -------------------------------------------------------------------

def _pose(x, y, yaw):
    from types import SimpleNamespace
    return SimpleNamespace(x=x, y=y, z=0.0, qw=math.cos(yaw / 2), qx=0.0, qy=0.0, qz=math.sin(yaw / 2))


def test_world_cuboid_round_trips_through_a_scan_pose():
    from services.lidar.aggregate.scene_frame import (
        ego_cuboid_to_world,
        pose_matrix,
        world_cuboid_to_ego,
    )

    t = pose_matrix(_pose(10.0, 5.0, 0.7))
    c, y = world_cuboid_to_ego([20.0, 12.0, -0.9], 1.2, t)
    back, yb = ego_cuboid_to_world(c, y, t)
    assert back == pytest.approx([20.0, 12.0, -0.9], abs=1e-3) and yb == pytest.approx(1.2, abs=1e-4)
    # straight ahead of a vehicle facing +y lands on +x in its ego frame
    c2, y2 = world_cuboid_to_ego([0.0, 10.0, 0.0], math.pi / 2, pose_matrix(_pose(0.0, 0.0, math.pi / 2)))
    assert c2[0] == pytest.approx(10.0, abs=1e-3) and abs(c2[1]) < 1e-3 and abs(y2) < 1e-4


def test_local_ground_and_box_support():
    from services.lidar.aggregate.label_propagate import local_ground_z, points_in_box_count

    rng = np.random.default_rng(2)
    ground = np.stack([rng.uniform(0, 40, 6000), rng.uniform(-10, 10, 6000), np.full(6000, -1.73)], 1)
    car = np.stack([rng.uniform(18, 22, 500), rng.uniform(2, 4, 500), rng.uniform(-1.6, -0.3, 500)], 1)
    xyz = np.vstack([ground, car])
    gz = local_ground_z(xyz, [20, 3, 0.0], [4.2, 1.9, 1.5], 0.0)
    assert gz == pytest.approx(-1.73, abs=0.02)
    assert points_in_box_count(xyz, [20, 3, -1.73 + 0.75], [4.2, 1.9, 1.5], 0.0) >= 450
    assert local_ground_z(xyz[:5], [20, 3, 0.0], [4, 2, 1.5], 0.0) is None


def test_cuboid_image_box_clips_at_the_near_plane():
    from services.calibration.resolve import Calibration
    from services.lidar.boxes import cuboid_image_box

    cal = Calibration(cam_id="c", model="pinhole", fx=700, fy=700, cx=620, cy=190, rpy_deg=(0, 0, 0), xyz_m=(0, 0, 0))
    box, why = cuboid_image_box([15, 0, 0], [4, 2, 1.5], 0.0, "c", 1242, 375, calib=cal)
    assert why == "ok" and box[0] < 620 < box[2]
    # a car alongside, half of it behind the lens: the box runs to the image edge instead of shrinking
    side, why = cuboid_image_box([0.5, -2.5, 0], [6, 2, 1.5], 0.0, "c", 1242, 375, calib=cal)
    assert why == "ok" and side[2] == pytest.approx(1242)
    assert cuboid_image_box([-10, 0, 0], [4, 2, 1.5], 0.0, "c", 1242, 375, calib=cal)[0] is None


# ---- point labels ------------------------------------------------------------------------------------------

def test_stride_and_subset_merge_keep_unseen_points():
    from services.lidar.segment3d.human import merge_subset, stride_for

    assert stride_for(120_000) == 1 and stride_for(1_000_000, 400_000) == 3
    base = np.full(10, 7, np.int32)
    out = merge_subset(base, np.array([1, 2, 3, 4], np.int32), 3)
    assert out.tolist() == [1, 7, 7, 2, 7, 7, 3, 7, 7, 4]
    with pytest.raises(ValueError):
        merge_subset(base, np.arange(5), 3)


def test_segmenter_classes_map_to_the_ontology():
    from services.autolabel.ontology import get_ontology
    from services.lidar.segment3d.semantic import segmenter_class_map

    onto = get_ontology()
    lut = segmenter_class_map(["road", "car", "vegetation", "unicorn"])
    assert lut[0] == onto.by_name("road").id and lut[2] == onto.by_name("vegetation").id
    assert lut[1] > 0 and lut[3] == -1


# ---- occupancy deltas --------------------------------------------------------------------------------------

def test_occupancy_ops_merge_and_transfer():
    from services.lidar.occupancy_edit import FREE, OCCUPIED, apply_ops, merge, transfer

    dims = (10, 10, 4)
    base = np.array([[1, 1, 1], [2, 2, 1], [5, 5, 0]], np.int32)
    flow = np.array([[1, 0, 0], [0, 0, 0], [0, 2, 0]], np.float32)
    dv, dval, touched = apply_ops(np.zeros((0, 3), np.int32), np.zeros(0, np.int8), [
        {"kind": "paint", "value": "free", "voxels": [[1, 1, 1]]},
        {"kind": "box", "value": "occupied", "lo": [7, 7, 0], "hi": [8, 8, 1]},
        {"kind": "paint", "value": "occupied", "voxels": [[5, 5, 0], [99, 0, 0]]},
    ], dims)
    assert touched == 1 + 8 + 1 and int((dval == OCCUPIED).sum()) == 9 and int((dval == FREE).sum()) == 1
    vox, f, edited = merge(base, flow, dv, dval, dims)
    keys = {tuple(v) for v in vox.tolist()}
    assert (1, 1, 1) not in keys and (2, 2, 1) in keys and (7, 8, 1) in keys and len(keys) == 1 + 9
    assert f[[tuple(v) for v in vox.tolist()].index((5, 5, 0))].tolist() == [0, 2, 0]   # re-confirmed keeps flow
    assert int(edited.sum()) == 9
    dv2, dval2, _ = apply_ops(dv, dval, [{"kind": "box", "value": "reset", "lo": [0, 0, 0], "hi": [9, 9, 3]}], dims)
    assert len(dv2) == 0 and len(dval2) == 0
    # one voxel (0.5 m) of world motion moves the edit one index back in the next grid
    tv, tval = transfer(np.array([[4, 4, 1]], np.int32), np.array([1], np.int8), [0, 0, 0], 0.5, [0.5, 0, 0], 0.5, dims)
    assert tv.tolist() == [[3, 4, 1]] and tval.tolist() == [1]
    with pytest.raises(ValueError):
        apply_ops(dv, dval, [{"kind": "paint", "value": "maybe", "voxels": []}], dims)


# ---- database ----------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


VELO_H = 1.73


def _scan(world_box_center, pose_xy, yaw, rng) -> np.ndarray:
    """A synthetic scan in the ego (Velodyne) frame: ground at -1.73 plus a 4.2 x 1.9 x 1.5 static box."""
    ground = np.stack([rng.uniform(-10, 50, 8000), rng.uniform(-20, 20, 8000), rng.normal(-VELO_H, 0.01, 8000)], 1)
    c, s = math.cos(yaw), math.sin(yaw)
    rel = np.asarray(world_box_center[:2]) - np.asarray(pose_xy)
    ex, ey = c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]
    car = np.stack([rng.uniform(-2.1, 2.1, 800), rng.uniform(-0.95, 0.95, 800), rng.uniform(0, 1.5, 800)], 1)
    car[:, 0], car[:, 1] = ex + car[:, 0], ey + car[:, 1]
    car[:, 2] += -VELO_H
    return np.vstack([ground, car]).astype(np.float32)


def _seed_scene(n: int = 5) -> dict:
    """A session with n LiDAR scans, an ego pose per scan (moving +x, turning slightly), a camera frame per
    scan and a forward camera calibration, observing one static box at world (22, 3)."""
    import cv2

    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import CameraCalibration, EgoPose, Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.normalize import Cloud
    from services.lidar.ingest.store import store_cloud

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    rng = np.random.default_rng(7)

    async def go():
        sid, t0 = uuid.uuid4(), now_ns()
        out = {"session_id": sid, "clouds": [], "frames": [], "ts": []}
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="SCENE3D-T", start_ts_ns=t0, end_ts_ns=t0 + 10**9,
                             city="Karlsruhe", ontology_version=onto.version,
                             sensors={"ego_frame": "velodyne", "lidar_height_m": VELO_H}))
            await db.flush()
            db.add(CameraCalibration(session_id=sid, cam_id="cam_front", model="pinhole", fx=700.0, fy=700.0,
                                     cx=620.0, cy=187.0, dist=[], ref_width=1242, rpy_deg=[0.0, 0.0, 0.0],
                                     xyz_m=[0.27, 0.0, -0.08], source="dataset", quality=0.95))
            ok, jpg = cv2.imencode(".jpg", np.full((375, 1242, 3), 80, np.uint8))
            for i in range(n):
                ts = t0 + i * 100_000_000
                fid = uuid.uuid4()
                uri = store.put_bytes(f"frames/{sid}/cam_front/{ts}.jpg", jpg.tobytes(), "image/jpeg")
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_front", img_uri=uri, width=1242,
                             height=375, quality=0.9, scene={}))
                out["frames"].append(str(fid))
                out["ts"].append(ts)
            await db.commit()
            for i, ts in enumerate(out["ts"]):
                x, yaw = 2.0 * i, 0.03 * i
                half = yaw / 2
                db.add(EgoPose(session_id=sid, ts_ns=ts, frame_id=uuid.UUID(out["frames"][i]), x=x, y=0.0, z=0.0,
                               qw=math.cos(half), qx=0.0, qy=0.0, qz=math.sin(half), source="gnss_imu",
                               quality=0.95, measured=True))
            await db.commit()
        for i, ts in enumerate(out["ts"]):
            xyz = _scan((22.0, 3.0), (2.0 * i, 0.0), 0.03 * i, rng)
            res = await store_cloud(Cloud(xyz=xyz, intensity=np.full(len(xyz), 0.5, np.float32), ts_ns=ts,
                                          source="lidar"), sid, source="lidar")
            out["clouds"].append(res["cloud_id"])
        return out

    return _run(go())


@pytest.mark.db
def test_label_once_places_one_box_on_every_scan_and_every_camera():
    from sqlalchemy import select

    from db.models import Object, Object3D, Track3D
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.aggregate.label_propagate import propagate_scene_label

    s = _seed_scene(5)
    sedan = get_ontology().by_name("sedan").id
    # drawn a little high on purpose: the per-scan ground snap must bring it down to the road
    res = _run(propagate_scene_label(s["session_id"], [22.0, 3.0, 0.0], [4.2, 1.9, 1.5], 0.0, sedan))
    assert res["scans"] == 5 and res["boxes_2d"] == 5 and res["ground_snapped"] == 5
    assert res["with_points"] == 5 and res["tracks_2d"].keys() == {"cam_front"}

    async def rows():
        async with get_sessionmaker()() as db:
            tr = await db.get(Track3D, uuid.UUID(res["track_3d_id"]))
            o3 = (await db.execute(select(Object3D).where(Object3D.track_3d_id == tr.track_3d_id))).scalars().all()
            o2 = (await db.execute(select(Object).where(Object.track_id == uuid.UUID(res["tracks_2d"]["cam_front"]))
                                   )).scalars().all()
            return tr, o3, o2

    tr, o3, o2 = _run(rows())
    assert tr.trajectory["static"] is True and str(tr.track_id) == res["tracks_2d"]["cam_front"]
    for o in o3:
        assert o.state == "review" and o.source == "human" and o.attrs["n_points"] >= 700
        assert o.center[2] == pytest.approx(-VELO_H + 0.75, abs=0.05)
        assert o.object_id is not None
    # the box sits where the world says it is in each scan's own frame: x shrinks as the ego drives toward it
    xs = sorted((o.center[0] for o in o3), reverse=True)
    assert xs[0] - xs[-1] > 6
    assert all(o.provenance["track_3d_id"] == res["track_3d_id"] and o.state == "review" for o in o2)
    assert all(0 <= o.bbox[0] < o.bbox[2] <= 1242 for o in o2)


@pytest.mark.db
def test_label_once_job_and_review_routes():
    from _authutil import auth_headers

    from services.api.main import app
    from services.autolabel.ontology import get_ontology
    from services.tools.registry import run_job

    s = _seed_scene(3)
    sid = str(s["session_id"])
    sedan = get_ontology().by_name("sedan").id
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        r = c.get(f"/api/tools/scene3d/sessions/{sid}/scene/poses", headers=ann)
        assert r.status_code == 200, r.text
        anchor = r.json()["anchor"]
        assert len(r.json()["poses"]) == 3
        pts = c.get(f"/api/tools/scene3d/sessions/{sid}/scene/points?every=1", headers=ann)
        assert pts.status_code == 200 and int(pts.headers["X-Scans"]) == 3
        assert len(pts.content) == 16 * int(pts.headers["X-Point-Count"])
        body = {"center": [22.0 - anchor[0], 3.0 - anchor[1], 0.0], "dims": [4.2, 1.9, 1.5], "yaw": 0.0,
                "class_id": sedan}
        assert c.post(f"/api/tools/scene3d/sessions/{sid}/label-once", headers=ann,
                      json={**body, "class_id": 99999}).status_code == 422

    # enqueue without starting, then run it here (the CPU kind would otherwise start in the API loop)
    async def enq():
        from db.session import get_sessionmaker
        from services.tools.registry import enqueue

        async with get_sessionmaker()() as db:
            job = await enqueue(db, "scene3d_label_once",
                                {"session_id": sid, "center": [22.0, 3.0, 0.0], "dims": [4.2, 1.9, 1.5],
                                 "yaw": 0.0, "class_id": sedan}, session_id=s["session_id"], start=False)
            return job.job_id

    jid = _run(enq())
    assert _run(run_job(jid)) == "done"

    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        tracks = c.get(f"/api/tools/scene3d/sessions/{sid}/label-once", headers=ann).json()["tracks"]
        assert len(tracks) == 1 and tracks[0]["states"] == {"review": 3}
        tid = tracks[0]["track_3d_id"]
        rv = c.get(f"/api/tools/scene3d/tracks3d/{tid}/review", headers=ann).json()
        assert rv["n"] == 3 and all(len(x["boxes"]) == 1 for x in rv["scans"])
        assert c.post(f"/api/tools/scene3d/tracks3d/{tid}/review", headers=ann,
                      json={"action": "accept"}).status_code == 403
        first = rv["scans"][0]["object_3d_id"]
        r = c.post(f"/api/tools/scene3d/tracks3d/{tid}/review", headers=rev,
                   json={"action": "reject", "object_3d_ids": [first]})
        assert r.status_code == 200, r.text
        assert r.json()["updated_3d"] == 1 and r.json()["updated_2d"] == 1
        r = c.post(f"/api/tools/scene3d/tracks3d/{tid}/review", headers=rev, json={"action": "accept"})
        assert r.json()["updated_3d"] == 3
        rv = c.get(f"/api/tools/scene3d/tracks3d/{tid}/review", headers=ann).json()
        assert {x["state"] for x in rv["scans"]} == {"accepted"}
        assert {b["state"] for x in rv["scans"] for b in x["boxes"]} == {"accepted"}


@pytest.mark.db
def test_point_labels_versioned_and_helpers():
    from _authutil import auth_headers

    from services.api.main import app
    from services.autolabel.ontology import get_ontology

    s = _seed_scene(1)
    cid = s["clouds"][0]
    onto = get_ontology()
    road, sedan = onto.by_name("road").id, onto.by_name("sedan").id
    with TestClient(app) as c:
        h = auth_headers("annotator")
        r = c.get(f"/api/tools/scene3d/clouds/{cid}/points?max=5000", headers=h)
        assert r.status_code == 200
        stride, n_sub, total = int(r.headers["X-Stride"]), int(r.headers["X-Point-Count"]), int(r.headers["X-Total"])
        assert stride == 2 and n_sub == len(range(0, total, stride))
        xyz = np.frombuffer(r.content, np.float32).reshape(-1, 4)[:, :3]

        g = c.post(f"/api/tools/scene3d/clouds/{cid}/assist/ground?stride={stride}", headers=h)
        gm = np.frombuffer(g.content, np.uint8).astype(bool)
        assert len(gm) == n_sub and gm.mean() > 0.85 and np.all(xyz[gm, 2] < -1.5)

        car_pt = xyz[~gm][np.argmax(xyz[~gm, 2])].tolist()
        cl = c.post(f"/api/tools/scene3d/clouds/{cid}/assist/cluster?stride={stride}", headers=h,
                    json={"point": car_pt, "eps": 0.6})
        cm = np.frombuffer(cl.content, np.uint8).astype(bool)
        # the box's lowest few centimetres sit inside the ground band; the cluster is the rest of it
        assert cl.status_code == 200 and cm.sum() > 300 and (cm & gm).sum() < 0.1 * cm.sum()

        labels = np.full(n_sub, -1, np.int16)
        labels[gm] = road
        labels[cm] = sedan
        put = c.put(f"/api/tools/scene3d/clouds/{cid}/labels?stride={stride}", headers=h, content=labels.tobytes())
        assert put.status_code == 200, put.text
        v1 = put.json()
        assert v1["version"] == 1 and v1["labelled"] == int((labels >= 0).sum())   # unseen points stay -1
        assert c.put(f"/api/tools/scene3d/clouds/{cid}/labels?stride={stride}", headers=h,
                     content=labels[:-1].tobytes()).status_code == 422

        # a second save at full resolution on top of v1: points the browser relabels change, the rest carry over
        full = c.get(f"/api/tools/scene3d/clouds/{cid}/labels?stride=1", headers=h)
        lab = np.frombuffer(full.content, np.int16).copy()
        assert full.headers["X-Seg-Source"] == "human" and (lab[::2] == labels).all()
        lab[lab == sedan] = onto.by_name("suv").id
        put2 = c.put(f"/api/tools/scene3d/clouds/{cid}/labels?stride=1&base_seg_id={v1['seg_id']}", headers=h,
                     content=lab.tobytes())
        assert put2.json()["version"] == 2
        vers = c.get(f"/api/tools/scene3d/clouds/{cid}/segmentations", headers=h).json()["versions"]
        assert [v["version"] for v in vers if v["source"] == "human"] == [2, 1]
        old = c.get(f"/api/tools/scene3d/clouds/{cid}/labels?stride={stride}&seg_id={v1['seg_id']}", headers=h)
        assert (np.frombuffer(old.content, np.int16) == labels).all()          # v1 is unchanged

        cub = c.post(f"/api/tools/scene3d/clouds/{cid}/assist/from-cuboids?stride=1", headers=h)
        assert cub.status_code == 200 and int(cub.headers["X-Selected"]) > 0     # the ground at least
        st = c.get("/api/tools/scene3d/ptv3/status", headers=h).json()
        assert "available" in st and "reason" in st


@pytest.mark.db
def test_occupancy_edit_layer_never_touches_the_generated_grid():
    from _authutil import auth_headers

    from core.storage import get_object_store
    from db.models import OccupancyGrid
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.lidar.occupancy4d import build_occupancy_window

    s = _seed_scene(3)
    sid = s["session_id"]
    res = _run(build_occupancy_window(sid))
    assert res["grids"] == 3 and res["grids_placed_by_pose"] == 3

    with TestClient(app) as c:
        h = auth_headers("annotator")
        grids = c.get(f"/api/tools/scene3d/sessions/{sid}/occupancy", headers=h).json()["grids"]
        assert len(grids) == 3 and not any(g["edited"] for g in grids)
        g0 = grids[0]["grid_id"]

        async def base_blob():
            async with get_sessionmaker()() as db:
                row = await db.get(OccupancyGrid, uuid.UUID(g0))
                return row.grid_uri, get_object_store().get_bytes(row.grid_uri)

        uri0, blob0 = _run(base_blob())
        before = c.get(f"/api/tools/scene3d/occupancy/{g0}", headers=h).json()
        v = before["voxels"][0]
        r = c.post(f"/api/tools/scene3d/occupancy/{g0}/edit", headers=h, json={"ops": [
            {"kind": "paint", "value": "free", "voxels": [v]},
            {"kind": "box", "value": "occupied", "lo": [100, 50, 3], "hi": [101, 51, 3]}]})
        assert r.status_code == 200, r.text
        assert r.json()["occupied_after"] == before["total"] - 1 + 4
        after = c.get(f"/api/tools/scene3d/occupancy/{g0}", headers=h).json()
        assert v not in after["voxels"] and [100, 50, 3] in after["voxels"] and len(after["set_occupied"]) == 4
        assert _run(base_blob()) == (uri0, blob0)                    # the generated grid is byte-identical

        p = c.post(f"/api/tools/scene3d/occupancy/{g0}/propagate", headers=h, json={"count": 5})
        assert p.status_code == 200, p.text
        assert p.json()["propagated_to"] == 2
        g1 = grids[1]["grid_id"]
        nxt = c.get(f"/api/tools/scene3d/occupancy/{g1}", headers=h).json()
        # the ego moved 2 m (4 voxels) forward and the window with it, so the painted box sits 4 voxels back
        assert [96, 50, 3] in nxt["set_occupied"]
        listed = c.get(f"/api/tools/scene3d/sessions/{sid}/occupancy", headers=h).json()["grids"]
        assert [g["edited"] for g in listed] == [True, True, True]
        assert c.post(f"/api/tools/scene3d/sessions/{sid}/occupancy/build", headers=h,
                      json={"mode": "lidar"}).status_code == 403


@pytest.mark.db
def test_a_camera_lift_does_not_overwrite_the_lidar_scan_at_the_same_instant():
    from core.timebase import now_ns
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.normalize import Cloud
    from services.lidar.ingest.store import load_cloud, store_cloud

    async def go():
        sid, ts = uuid.uuid4(), now_ns()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="SCENE3D-K", start_ts_ns=ts, end_ts_ns=ts + 1, city="BLR",
                             sensors={}, ontology_version=get_ontology().version))
            await db.commit()
        a = await store_cloud(Cloud(xyz=np.ones((10, 3), np.float32), intensity=np.ones(10, np.float32), ts_ns=ts,
                                    source="lidar"), sid, source="lidar")
        b = await store_cloud(Cloud(xyz=np.zeros((7, 3), np.float32), intensity=np.zeros(7, np.float32), ts_ns=ts,
                                    source="pseudo"), sid, source="pseudo")
        return a["cloud_uri"], b["cloud_uri"]

    ua, ub = _run(go())
    assert ua != ub and load_cloud(ua).n == 10 and load_cloud(ub).n == 7


_KITTI_SCAN = Path(".scratch/demo/kitti/2011_09_26/2011_09_26_drive_0015_sync/velodyne_points/data/0000000010.bin")


@pytest.mark.gpu
@pytest.mark.skipif(not _KITTI_SCAN.exists(), reason="KITTI raw drive 0015 is not on disk")
def test_ptv3_segments_a_real_kitti_scan():
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.readers import read_kitti_bin
    from services.lidar.segment3d import ptv3_infer
    from services.lidar.segment3d.semantic import segment_ptv3

    ok, why = ptv3_infer.available()
    if not ok:
        pytest.skip(why)
    cloud = read_kitti_bin(_KITTI_SCAN.read_bytes())
    try:
        res = segment_ptv3(cloud)
    finally:
        ptv3_infer.release()
    onto = get_ontology()
    sem = res["semantic"]
    low = cloud.xyz[:, 2] < -1.5
    ground_like = {onto.by_name(n).id for n in ("road", "sidewalk", "green_belt", "non_drivable_fallback", "parking_area")}
    assert len(sem) == cloud.n and res["low_conf_frac"] < 0.2
    assert np.isin(sem[low], list(ground_like)).mean() > 0.8      # the road surface is called a ground class
    assert onto.by_name("vehicle_fallback").id in res["classes_present"]
