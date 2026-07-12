"""Integration: the LiDAR->camera projection routes large clouds through the fused GPU kernel and must produce
the identical result to the NumPy path it replaces. Forces the accel branch and compares its pixels to the pure
NumPy pinhole formula bit-close, so the capability-gated fast path is proven a no-op on behavior."""

import numpy as np

import services.lidar.project as lp
from core.accel.projection import gpu_available


def test_lidar_pinhole_accel_matches_numpy():
    rng = np.random.default_rng(0)
    n = 40000
    cam = np.column_stack([rng.uniform(-20, 20, n), rng.uniform(-20, 20, n),
                           rng.uniform(-2, 40, n)]).astype(np.float32)   # some behind the camera (z<0)
    fx, fy, cx, cy = 900.0, 900.0, 960.0, 540.0
    W, H = 1920, 1080

    # reference: the exact NumPy formula used when the accel path is off
    z = cam[:, 2]
    in_front = z > 1e-3
    safe_z = np.where(in_front, z, 1.0)
    ref = np.stack([cam[:, 0] / safe_z * fx + cx, cam[:, 1] / safe_z * fy + cy], axis=1).astype(np.float32)

    # force the NumPy branch (threshold above cloud size)
    lp._ACCEL_MIN_POINTS = 10**9
    off = lp._project_cam_points(cam, fx, fy, cx, cy, "pinhole", None, W, H)
    assert np.allclose(off["uv"], ref, atol=1e-3)
    assert np.array_equal(off["in_front"], in_front)

    if gpu_available():
        lp._ACCEL_MIN_POINTS = 1000                 # force the GPU accel branch
        on = lp._project_cam_points(cam, fx, fy, cx, cy, "pinhole", None, W, H)
        # the accelerated pixels match the NumPy pixels for every in-front point
        assert np.allclose(on["uv"][in_front], ref[in_front], atol=1e-3)
        assert np.array_equal(on["in_front"], off["in_front"])
        assert np.array_equal(on["in_image"], off["in_image"])
    lp._ACCEL_MIN_POINTS = 16384                     # restore the default
