"""M-L1.4: cloud points project into the camera and back consistently. A pixel's ray passes through the ego
point that projected to it, and lift_pixel recovers the exact cloud point a pixel sees."""

from __future__ import annotations

import numpy as np

from services.lidar.ingest import Cloud
from services.lidar.project import (
    camera_ray_to_ego,
    ego_to_camera,
    lift_pixel,
    project_to_camera,
)

W, H = 1280, 960


def test_pinhole_project_then_unproject_is_consistent():
    # a point ahead of the front camera (10 m forward, 2 m left, 0.5 m up)
    p = np.array([[10.0, 2.0, 0.5]], dtype=np.float32)
    proj = project_to_camera(p, "cam_f", W, H)
    assert proj["in_front"][0] and proj["in_image"][0]
    u, v = proj["uv"][0]
    ray = camera_ray_to_ego(float(u), float(v), "cam_f", W, H)
    to_p = p[0] - ray["origin"]
    to_p = to_p / np.linalg.norm(to_p)
    assert np.allclose(to_p, ray["direction"], atol=1e-4)   # the pixel's ray points at the original point
    assert float((p[0] - ray["origin"]) @ ray["direction"]) > 0


def test_ego_to_camera_axes():
    # a point straight ahead at camera height projects to the image centre of the front camera
    p = np.array([[12.0, 0.0, 1.5]], dtype=np.float32)   # 1.5 m up == camera mount height
    cam = ego_to_camera(p, "cam_f")
    assert cam[0, 2] > 0                                   # in front of the camera
    proj = project_to_camera(p, "cam_f", W, H)
    assert abs(proj["uv"][0, 0] - W / 2) < 1.0 and abs(proj["uv"][0, 1] - H / 2) < 1.0


def test_points_behind_camera_are_not_in_front():
    p = np.array([[-10.0, 0.0, 1.5], [10.0, 0.0, 1.5]], dtype=np.float32)  # behind, ahead
    proj = project_to_camera(p, "cam_f", W, H)
    assert not proj["in_front"][0] and proj["in_front"][1]


def test_lift_pixel_recovers_cloud_point():
    rng = np.random.default_rng(0)
    target = np.array([8.0, -1.0, 0.7], dtype=np.float32)
    others = rng.uniform(-5, 30, (400, 3)).astype(np.float32)
    xyz = np.vstack([target[None], others])
    cloud = Cloud(xyz=xyz, intensity=np.ones(len(xyz), np.float32), ts_ns=1)
    uv = project_to_camera(target[None], "cam_f", W, H)["uv"][0]
    hit = lift_pixel(float(uv[0]), float(uv[1]), "cam_f", cloud, W, H, max_dist=0.5)
    assert hit is not None and hit["index"] == 0
    assert np.allclose(hit["point"], target, atol=1e-3)


def test_fisheye_side_camera_round_trips():
    # the wide surround cameras use the fisheye model; project then unproject within tolerance
    p = np.array([[6.0, 8.0, 0.4]], dtype=np.float32)     # to the left, where cam_l (yaw -90) looks
    proj = project_to_camera(p, "cam_l", W, H)
    assert proj["in_front"][0]
    u, v = proj["uv"][0]
    ray = camera_ray_to_ego(float(u), float(v), "cam_l", W, H)
    to_p = p[0] - ray["origin"]
    to_p = to_p / np.linalg.norm(to_p)
    assert np.allclose(to_p, ray["direction"], atol=1e-3)
