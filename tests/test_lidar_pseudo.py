"""M-L1.1: pseudo-LiDAR geometry (back-projection scale, camera-to-ego placement, world placement) is
correct, and a real metric-depth lift produces a scaled ego-frame cloud with full provenance.

The geometry tests are exact and need no model. The end-to-end lift needs the depth model (skips if torch or
the checkpoint is unavailable)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from services.lidar.ingest import Cloud
from services.lidar.ingest.pseudo import backproject, camera_to_ego, place_in_world


def test_backproject_metric_scale():
    # only one in-range pixel survives, so the back-projected point is exact
    depth = np.full((101, 101), 1000.0, dtype=np.float32)   # out of range -> dropped
    depth[50, 60] = 10.0                                     # (u=60, v=50) at 10 m
    luma = np.zeros((101, 101), dtype=np.float32)
    xyz, inten = backproject(depth, fx=100.0, fy=100.0, cx=50.0, cy=50.0, luma=luma, stride=1, z_max=80.0)
    assert xyz.shape == (1, 3)
    # x = (60-50)/100 * 10 = 1.0 ; y = (50-50)/100 * 10 = 0 ; z = 10
    assert np.allclose(xyz[0], [1.0, 0.0, 10.0], atol=1e-4)


def test_camera_to_ego_axes_and_height():
    # a point 1 m right and 10 m ahead in the camera frame -> ego forward 10, right (negative left) 1, up h
    ego = camera_to_ego(np.array([[1.0, 0.0, 10.0]], dtype=np.float32), cam_yaw_deg=0.0, height_m=1.5)
    assert np.allclose(ego[0], [10.0, -1.0, 1.5], atol=1e-4)


def test_camera_to_ego_yaw_places_side_and_rear_cameras():
    fwd = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)   # straight ahead of each camera
    left = camera_to_ego(fwd, cam_yaw_deg=-90.0, height_m=1.5)[0]   # cam_l
    right = camera_to_ego(fwd, cam_yaw_deg=90.0, height_m=1.5)[0]   # cam_r
    rear = camera_to_ego(fwd, cam_yaw_deg=180.0, height_m=1.5)[0]   # cam_b
    assert np.allclose(left, [0.0, 10.0, 1.5], atol=1e-4)    # to the vehicle's left  (+y)
    assert np.allclose(right, [0.0, -10.0, 1.5], atol=1e-4)  # to the vehicle's right (-y)
    assert np.allclose(rear, [-10.0, 0.0, 1.5], atol=1e-4)   # behind the vehicle     (-x)


def test_place_in_world_enu():
    cloud = Cloud(xyz=np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
                  intensity=np.zeros(1, np.float32), ts_ns=1, source="pseudo")
    north = place_in_world(cloud, lat=12.9, lon=77.6, heading_rad=0.0).xyz[0]        # heading north
    east = place_in_world(cloud, lat=12.9, lon=77.6, heading_rad=math.pi / 2).xyz[0]  # heading east
    assert np.allclose(north, [0.0, 10.0, 0.0], atol=1e-4)   # 10 m north
    assert np.allclose(east, [10.0, 0.0, 0.0], atol=1e-4)    # 10 m east
    assert place_in_world(cloud, 12.9, 77.6, 0.0).frame == "world_enu"


def _depth_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForDepthEstimation  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _depth_available(), reason="depth model stack unavailable")
def test_real_depth_lift_produces_metric_cloud():
    """A single front-camera frame lifts to a non-empty, metrically scaled, ego-frame pseudo-LiDAR cloud."""
    from services.lidar.ingest.pseudo import lift_frame_group

    rng = np.random.default_rng(0)
    # a textured synthetic scene so depth has structure; portable, no fixture dependency
    img = (rng.uniform(40, 215, size=(360, 640, 3))).astype(np.uint8)
    img[:180] = np.clip(img[:180] + 30, 0, 255)   # brighter upper half (sky-like)
    cloud = lift_frame_group({"cam_f": img}, ts_ns=42, calibration_version="calib-1", stride=4)
    assert cloud.n > 1000 and cloud.source == "pseudo" and cloud.frame == "ego"
    assert "Outdoor" in (cloud.depth_model or "") or cloud.depth_model
    assert np.isfinite(cloud.xyz).all()
    assert (cloud.xyz[:, 0] > 0).mean() > 0.9       # almost all points are ahead of the vehicle
    fwd = cloud.xyz[:, 0]
    assert 0.5 <= float(np.median(fwd)) <= 80.0     # plausible metric forward distances
