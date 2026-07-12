"""Fused projection kernel (Tier 1) verification: the batched GPU/torch path must agree bit-close with the
established references it accelerates. Pinhole is checked against core.geometry.project_points (the NumPy
reference the kernel exists to speed up) and against the NumPy backend of the kernel itself; fisheye is checked
against cv2.fisheye.projectPoints. A session-scale timing is printed so the win is measurable."""

import time

import numpy as np
import pytest

from core.accel.projection import gpu_available, project_world_batch
from core.geometry import project_points, se3


def _rand_pose(rng):
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=rng).as_matrix()
    t = rng.uniform(-3, 3, size=3)
    return se3(R, t)


def _K(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def test_pinhole_matches_geometry_reference():
    rng = np.random.default_rng(7)
    M, C = 200, 6
    pts_world = rng.uniform(-8, 8, size=(M, 3))
    Ts = np.stack([_rand_pose(rng) for _ in range(C)])
    Ks = np.stack([_K(1000 + 5 * c, 1000 + 5 * c, 960, 540) for c in range(C)])
    dists = np.stack([np.array([-0.32, 0.11, 0.001, -0.0008, 0.02]) for _ in range(C)])

    uv, valid, _ = project_world_batch(pts_world, Ts, Ks, dists, model="pinhole")

    # per-camera reference: transform world->cam with the same pose, then project_points
    for c in range(C):
        pts_cam = pts_world @ Ts[c][:3, :3].T + Ts[c][:3, 3]
        ref_uv, ref_valid = project_points(pts_cam, Ks[c], dists[c])
        # compare only points in front of the camera (behind-camera pixels are undefined in both)
        m = ref_valid
        assert np.array_equal(valid[c], ref_valid)
        assert np.allclose(uv[c][m], ref_uv[m], atol=1e-6), f"camera {c} pinhole mismatch"


def test_fisheye_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    M = 150
    pts_world = np.column_stack([rng.uniform(-3, 3, M), rng.uniform(-3, 3, M), rng.uniform(2, 12, M)])
    T = se3(np.eye(3), np.zeros(3))                       # camera at world origin, so world==camera
    K = _K(700, 700, 960, 600)
    dist = np.array([-0.08, 0.01, -0.002, 0.0004])       # Kannala-Brandt k1..k4

    uv, _, _ = project_world_batch(pts_world, T[None], K[None], dist[None], model="fisheye")

    ref, _ = cv2.fisheye.projectPoints(pts_world.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                       K, dist.reshape(4, 1))
    ref = ref.reshape(-1, 2)
    assert np.allclose(uv[0], ref, atol=1e-4), "fisheye mismatch vs cv2.fisheye"


def test_gpu_matches_cpu_and_is_measurable():
    rng = np.random.default_rng(3)
    M, C = 2000, 6                                        # session-scale: 2000 boxes x 6 cameras
    pts_world = rng.uniform(-10, 10, size=(M, 3))
    Ts = np.stack([_rand_pose(rng) for _ in range(C)])
    Ks = np.stack([_K() for _ in range(C)])
    dists = np.stack([np.array([-0.3, 0.1, 0.0, 0.0, 0.0]) for _ in range(C)])
    wh = np.stack([np.array([1920.0, 1080.0]) for _ in range(C)])

    uv_cpu, valid_cpu, ib_cpu = project_world_batch(pts_world, Ts, Ks, dists, wh, device="cpu")
    # the reference NumPy path is internally consistent (bounds subset of valid)
    assert np.all(ib_cpu <= valid_cpu)

    if gpu_available():
        uv_gpu, valid_gpu, ib_gpu = project_world_batch(pts_world, Ts, Ks, dists, wh, device="cuda")
        assert np.array_equal(valid_cpu, valid_gpu) and np.array_equal(ib_cpu, ib_gpu)
        assert np.allclose(uv_cpu[valid_cpu], uv_gpu[valid_gpu], atol=1e-6)

        # session scale: many boxes across the rig. Big batch so the GPU is actually fed and the H2D/D2H
        # transfer amortizes.
        Mbig = 60000
        pts_big = rng.uniform(-10, 10, size=(Mbig, 3))
        for _ in range(3):
            project_world_batch(pts_big, Ts, Ks, dists, wh, device="cuda")
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            project_world_batch(pts_big, Ts, Ks, dists, wh, device="cuda")
        gpu_ms = (time.perf_counter() - t0) / n * 1000

        # baseline the kernel replaces: the per-camera, per-box NumPy-bound loop in the current code path
        t0 = time.perf_counter()
        for c in range(C):
            pc = pts_big @ Ts[c][:3, :3].T + Ts[c][:3, 3]
            for j in range(0, Mbig, 500):                 # chunked to keep the pure-loop cost realistic
                project_points(pc[j:j + 500], Ks[c], dists[c])
        loop_ms = (time.perf_counter() - t0) * 1000
        total = C * Mbig
        print(f"\nprojection {C}x{Mbig} = {total:,} points/pass: "
              f"GPU {gpu_ms:.2f} ms ({total / gpu_ms / 1000:.1f}M pts/ms) | "
              f"per-camera NumPy loop {loop_ms:.2f} ms | speedup {loop_ms / gpu_ms:.1f}x")
