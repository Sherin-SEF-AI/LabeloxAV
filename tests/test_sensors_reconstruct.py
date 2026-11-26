"""Scene reconstruction (#15): camera geometry, the point map, gsplat on the GPU, and place-then-propagate.

The geometry is checked against answers known in advance: a camera mounted forward sees the ego +x on its
optical axis, a top-down view maps a world point to the pixel its formula says, a point hidden behind another
takes no colour from that view. The end-to-end paths run on a generated session (frames, ego poses, LiDAR,
calibration) whose colours and layout are chosen so the result can be checked, not only produced.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from services.sensors.reconstruct.window import (
    WindowFrame,
    bev_view,
    camera_to_ego,
    colourise,
    frustum_mask,
    look_at,
    nearest_pose_index,
    novel_poses,
)


def _front_camera(eye=(0.0, 0.0, 1.5), w=320, h=240, f=200.0) -> WindowFrame:
    """A camera at `eye` looking along world +x (optical z = world x, optical x = world -y, optical y = -z)."""
    T = np.eye(4)
    T[:3, 0] = [0, -1, 0]
    T[:3, 1] = [0, 0, -1]
    T[:3, 2] = [1, 0, 0]
    T[:3, 3] = eye
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return WindowFrame(frame_id=uuid.uuid4(), ts_ns=0, cam_id="cam", img_uri="", width=w, height=h, K=K,
                       T_world_cam=T)


def test_calibration_chain_puts_the_optical_axis_along_ego_forward():
    from services.calibration.resolve import Calibration

    cal = Calibration(cam_id="cam_front", model="pinhole", fx=700, fy=700, cx=620, cy=187,
                      rpy_deg=(0.0, 0.0, 0.0), xyz_m=(1.5, 0.0, 1.4), source="dataset")
    T = camera_to_ego(cal.R(), cal.t())
    # the optical axis (0, 0, 1) in the camera is +x in the ego, starting at the mount
    p = T @ np.array([0, 0, 10.0, 1.0])
    assert p[:3].tolist() == pytest.approx([11.5, 0.0, 1.4], abs=1e-5)
    # and the inverse is what project_to_camera does: cam = (ego - t) @ R
    ego = np.array([[20.0, -3.0, 0.5]])
    cam = (ego - cal.t()) @ cal.R()
    back = (np.linalg.inv(T) @ np.r_[ego[0], 1.0])[:3]
    assert back.tolist() == pytest.approx(cam[0].tolist(), abs=1e-4)


def test_nearest_pose_respects_the_tolerance():
    ts = np.array([0, 100, 200], dtype=np.int64) * 1_000_000
    assert nearest_pose_index(ts, 110_000_000) == 1
    assert nearest_pose_index(ts, 290_000_000) is None            # 90 ms from the last: another instant
    assert nearest_pose_index(np.array([], dtype=np.int64), 5) is None


def test_novel_poses_are_where_they_say():
    f = _front_camera()
    nv = novel_poses(f, np.array([0, 0, 1.0]))
    assert nv["left_1.5m"][:3, 3].tolist() == pytest.approx([0, 1.5, 1.5])      # camera right is world -y
    assert nv["right_1.5m"][:3, 3].tolist() == pytest.approx([0, -1.5, 1.5])
    r = nv["raised_3m"]
    assert r[2, 3] == pytest.approx(4.5) and r[0, 3] == pytest.approx(-2.0)
    assert r[2, 2] < 0                                                           # looking down
    # look_at gives a proper rotation
    m = look_at(np.array([0, 0, 5.0]), np.array([10, 0, 0.0]), np.array([0, 0, 1.0]))
    assert np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-9) and np.linalg.det(m[:3, :3]) == pytest.approx(1)


def test_bev_view_maps_world_to_pixels_with_the_drive_pointing_up():
    pts = np.array([[0, 0, 0], [0, 40, 0], [10, 20, 2.0]])
    b = bev_view(pts, heading_xy=np.array([0.0, 1.0]), max_px=500, margin_m=5.0)
    T, K = b["T_world_cam"], b["K"]
    inv = np.linalg.inv(T)

    def px(p):
        c = inv[:3, :3] @ p + inv[:3, 3]
        return K[0, 0] * c[0] + K[0, 2], K[1, 1] * c[1] + K[1, 2], c[2]
    u0, v0, z0 = px(np.array([0, 0, 0.0]))
    u1, v1, _ = px(np.array([0, 40, 0.0]))
    assert v1 < v0 and u1 == pytest.approx(u0)              # further along the drive is higher on screen
    assert z0 > 0 and np.linalg.det(T[:3, :3]) == pytest.approx(1)
    assert 0 <= u0 <= b["width"] and 0 <= v0 <= b["height"] and b["ppm"] == pytest.approx(500 / 50)


def test_colourise_takes_colour_only_from_views_that_see_a_point():
    f = _front_camera()
    img = np.zeros((f.height, f.width, 3), np.uint8)
    img[:, : f.width // 2] = (255, 0, 0)          # left half red
    img[:, f.width // 2:] = (0, 0, 255)           # right half blue
    pts = np.array([
        [10.0, 2.0, 1.5],      # left of centre (world +y is image left): red
        [10.0, -2.0, 1.5],     # right: blue
        [20.0, 4.0, 1.5],      # directly behind the first on the same ray: hidden
        [-5.0, 0.0, 1.5],      # behind the camera: never seen
    ])
    rgb, seen = colourise(pts, [f], [img])
    assert seen.tolist() == [True, True, False, False]
    assert rgb[0].tolist() == pytest.approx([1, 0, 0]) and rgb[1].tolist() == pytest.approx([0, 0, 1])
    assert frustum_mask(pts, [f]).tolist() == [True, True, True, False]


def test_point_render_draws_the_nearest_point():
    from services.sensors.reconstruct.pointmap import render_points

    f = _front_camera(w=64, h=48, f=40.0)
    pts = np.array([[10.0, 0.0, 1.5], [5.0, 0.0, 1.5]])
    rgb = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    img = render_points(pts, rgb, f.T_world_cam, f.K, f.width, f.height, downscale=1)
    assert img[24, 32].tolist() == pytest.approx([0, 1, 0])        # the near green point wins its pixel
    assert img[0, 0].tolist() == pytest.approx([0.6, 0.65, 0.7])   # nothing reached the corner


def test_holdout_split_and_keyframes():
    from services.sensors.reconstruct.gsplat_train import split_holdout, to_ply
    from services.sensors.reconstruct.run import keyframe_indices

    train, held = split_holdout(50)
    assert held == [4, 12, 20, 28, 36, 44] and not set(train) & set(held) and len(train) + len(held) == 50
    assert split_holdout(5) == ([0, 1, 2, 3, 4], [])
    assert keyframe_indices(50, 6, held=[4]) == [0, 4, 10, 20, 29, 39, 49]
    g = {"means": np.zeros((3, 3)), "scales": np.zeros((3, 3)), "quats": np.tile([1, 0, 0, 0.0], (3, 1)),
         "opacities": np.zeros(3), "sh0": np.zeros((3, 1, 3)), "shN": np.zeros((3, 8, 3))}
    ply = to_ply(g)
    head, body = ply.split(b"end_header\n")
    assert b"element vertex 3" in head and head.count(b"property float") == 3 + 3 + 3 + 24 + 1 + 3 + 4
    assert len(body) == 3 * 41 * 4


@pytest.mark.gpu
def test_gsplat_learns_a_coloured_plane_and_renders_a_held_out_view():
    """Splats initialised on a two-coloured wall, trained on views rendered from that wall, reproduce a view
    they were not trained on."""
    pytest.importorskip("gsplat")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from services.sensors.reconstruct.gsplat_train import render_views, train_splats
    from services.sensors.reconstruct.pointmap import render_points

    ys, zs = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(0, 3, 50))
    wall = np.stack([np.full(ys.size, 8.0), ys.ravel(), zs.ravel()], axis=1)
    rgb = np.where(wall[:, 1:2] > 0, [[0.9, 0.2, 0.1]], [[0.1, 0.3, 0.9]]).astype(np.float32)
    frames = [_front_camera(eye=(0.0, y, 1.5), w=96, h=72, f=60.0) for y in np.linspace(-1, 1, 9)]
    imgs = [(render_points(wall, rgb, f.T_world_cam, f.K, f.width, f.height, base_px=6) * 255).astype(np.uint8)
            for f in frames]
    vm = np.stack([np.linalg.inv(f.T_world_cam) for f in frames])
    Ks = np.stack([f.K for f in frames])
    res = train_splats(imgs, vm, Ks, wall.astype(np.float32), rgb, iters=300, sh_degree=0,
                       background=np.array([0.6, 0.65, 0.7]))
    m = res["metrics"]
    assert m["held_out_frames"] == 1 and m["psnr_held_out"] > 18 and m["gaussians"] > 0
    out = render_views(res["gaussians"], vm[4:5], Ks[4:5], 96, 72, background=np.array([0.6, 0.65, 0.7]))[0]
    # left of the image is world +y: red
    assert out[40, 20, 0] > out[40, 20, 2] and out[40, 76, 2] > out[40, 76, 0]
    torch.cuda.empty_cache()


# ---- database ----------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed_drive(n: int = 10) -> dict:
    """A session driving +x at 10 m/s past a wall at y = +6, with a camera frame, an ego pose and a LiDAR scan
    every 100 ms, and a forward camera calibration. Every image is the same orange, so every coloured point must
    come out orange."""
    import cv2

    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import CameraCalibration, EgoPose, Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.lidar.ingest.normalize import Cloud
    from services.lidar.ingest.store import store_cloud

    store = get_object_store()
    store.ensure_bucket()
    rng = np.random.default_rng(3)

    async def go():
        sid, t0 = uuid.uuid4(), now_ns()
        out = {"session_id": sid, "frames": [], "ts": []}
        ok, jpg = cv2.imencode(".jpg", np.full((240, 320, 3), (0, 128, 255), np.uint8))   # BGR orange
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RECON-T", start_ts_ns=t0, end_ts_ns=t0 + 10**10,
                             ontology_version=get_ontology().version, sensors={}))
            await db.flush()
            db.add(CameraCalibration(session_id=sid, cam_id="cam_front", model="pinhole", fx=200.0, fy=200.0,
                                     cx=160.0, cy=120.0, dist=[], ref_width=320, rpy_deg=[0.0, 0.0, 0.0],
                                     xyz_m=[1.5, 0.0, 1.4], source="dataset", quality=0.9))
            for i in range(n):
                ts = t0 + i * 100_000_000
                uri = store.put_bytes(f"frames/{sid}/cam_front/{ts}.jpg", jpg.tobytes(), "image/jpeg")
                f = Frame(session_id=sid, ts_ns=ts, cam_id="cam_front", img_uri=uri, width=320, height=240,
                          quality=1.0)
                db.add(f)
                await db.flush()
                db.add(EgoPose(session_id=sid, ts_ns=ts, frame_id=f.frame_id, x=1.0 * i, y=0.0, z=0.0, qw=1.0,
                               qx=0.0, qy=0.0, qz=0.0, source="gnss_imu", quality=0.9, measured=True))
                out["frames"].append(str(f.frame_id))
                out["ts"].append(ts)
            await db.commit()
        for i, ts in enumerate(out["ts"]):
            # ground ahead, and a wall at y = 6 from x = 0..40, in the ego frame of scan i (ego at x = i)
            g = np.stack([rng.uniform(2, 30, 3000), rng.uniform(-8, 8, 3000), np.zeros(3000)], 1)
            wall = np.stack([rng.uniform(0, 40, 3000) - i, np.full(3000, 6.0), rng.uniform(0, 3, 3000)], 1)
            xyz = np.vstack([g, wall]).astype(np.float32)
            await store_cloud(Cloud(xyz=xyz, intensity=np.full(len(xyz), 0.5, np.float32), ts_ns=ts,
                                    source="lidar"), sid, source="lidar")
        return out

    return _run(go())


@pytest.mark.db
def test_pointmap_reconstruction_end_to_end():
    from db.models import Reconstruction
    from db.session import get_sessionmaker
    from services.sensors.reconstruct.run import build_reconstruction

    s = _seed_drive(10)

    async def mk():
        async with get_sessionmaker()() as db:
            r = Reconstruction(session_id=s["session_id"], method="pointmap", params={"n_frames": 10})
            db.add(r)
            await db.commit()
            return r.recon_id

    rid = _run(mk())
    res = _run(build_reconstruction(rid, device="cpu"))
    assert res["method"] == "pointmap" and res["views"] == 6 * 4 + 1

    async def get():
        async with get_sessionmaker()() as db:
            return await db.get(Reconstruction, rid)

    r = _run(get())
    assert r.status == "done" and r.cam_id == "cam_front" and r.first_ts_ns == s["ts"][0]
    assert r.metrics["window"]["frames_posed"] == 10 and r.metrics["initial_points"] > 1000
    bev = [v for v in r.views if v["kind"] == "bev"]
    assert len(bev) == 1 and bev[0]["ortho"] and bev[0]["heading"] == pytest.approx([1.0, 0.0])
    assert sum(1 for v in r.views if v["kind"] == "camera") == 6
    # the camera frames are uniformly orange, so every map point is orange (RGB 255, 128, 0)
    from core.storage import get_object_store

    pts = np.frombuffer(get_object_store().get_bytes(r.artifacts["preview"]), np.float32).reshape(-1, 6)
    assert np.allclose(pts[:, 3:].mean(axis=0), [1.0, 0.5, 0.0], atol=0.04)


@pytest.mark.db
def test_reconstruct_routes_roles_and_place_goes_to_label_once():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from db.models import ToolJob
    from db.session import get_sessionmaker
    from services.api.main import app
    from services.autolabel.ontology import get_ontology
    from services.tools.registry import run_job

    s = _seed_drive(6)
    sid = str(s["session_id"])
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        sess = c.get("/api/tools/sensors/reconstruct/sessions", headers=ann).json()
        assert any(x["session_id"] == sid and x["cameras"][0]["cam_id"] == "cam_front" for x in sess["sessions"])
        # training splats is a reviewer action; the point map is not
        assert c.post("/api/tools/sensors/reconstruct", headers=ann,
                      json={"session_id": sid, "method": "gsplat"}).status_code == 403
        r = c.post("/api/tools/sensors/reconstruct", headers=ann,
                   json={"session_id": sid, "method": "pointmap", "n_frames": 6})
        assert r.status_code == 200, r.text
        rid, jid = r.json()["reconstruction"]["recon_id"], r.json()["job"]["job_id"]
        assert r.json()["job"]["kind"] == "sensors_pointmap"
    _run(run_job(uuid.UUID(jid)))
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        d = c.get(f"/api/tools/sensors/reconstructions/{rid}", headers=ann).json()
        assert d["status"] == "done" and d["n_views"] == len(d["views"]) and "uri" not in d["views"][0]
        img = c.get(f"/api/tools/sensors/reconstructions/{rid}/views/0.jpg", headers=ann)
        assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
        pv = c.get(f"/api/tools/sensors/reconstructions/{rid}/preview.bin", headers=ann)
        anchor = [float(v) for v in pv.headers["X-Anchor"].split(",")]
        assert len(pv.content) == 24 * int(pv.headers["X-Count"]) and len(anchor) == 3
        assert c.get(f"/api/tools/sensors/reconstructions/{rid}/splats.ply", headers=ann).status_code == 404
        sedan = get_ontology().by_name("sedan").id
        bad = c.post(f"/api/tools/sensors/reconstructions/{rid}/place", headers=ann,
                     json={"center": [10, 6, 1], "dims": [0.01, 1, 1], "yaw": 0, "class_id": sedan})
        assert bad.status_code == 422
        r = c.post(f"/api/tools/sensors/reconstructions/{rid}/place", headers=ann,
                   json={"center": [12.0, 3.0, 0.75], "dims": [4.2, 1.8, 1.5], "yaw": 0.0, "class_id": sedan})
        assert r.status_code == 200 and r.json()["kind"] == "scene3d_label_once"
        place_job = r.json()["job_id"]
        placed = c.get(f"/api/tools/sensors/reconstructions/{rid}/placed", headers=ann).json()["placed"]
        assert placed[0]["job_id"] == place_job and placed[0]["center"] == [12.0, 3.0, 0.75]
        assert c.delete(f"/api/tools/sensors/reconstructions/{rid}", headers=ann).status_code == 403

    async def job_params():
        async with get_sessionmaker()() as db:
            return (await db.get(ToolJob, uuid.UUID(place_job))).params

    p = _run(job_params())
    assert p["session_id"] == sid and p["recon_id"] == rid and p["project"] is True
    with TestClient(app) as c:
        rev = auth_headers("reviewer")
        assert c.delete(f"/api/tools/sensors/reconstructions/{rid}", headers=rev).status_code == 200
