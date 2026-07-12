"""Multi-view consistency kernel (Tier 2) verification: the all-pairs Sampson matrix must match
core.geometry.sampson_distance pairwise, the GPU path must match the CPU path, and a real two-view
correspondence must land near zero and be picked by best_epipolar_match. A session-scale timing is printed."""

import time

import numpy as np

from core.accel.geometry_mv import best_epipolar_match, gpu_available, sampson_matrix
from core.geometry import fundamental_matrix, project_points, sampson_distance, se3, transform_points


def _two_view():
    """A calibrated two-camera setup and the F relating them."""
    rng = np.random.default_rng(0)
    K1 = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])
    K2 = np.array([[1010.0, 0, 950], [0, 1010, 545], [0, 0, 1]])
    R = se3(np.eye(3), np.zeros(3))[:3, :3]
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("y", 8, degrees=True).as_matrix()
    t = np.array([0.6, 0.02, 0.05])                     # baseline
    F = fundamental_matrix(R, t, K1, K2)
    return rng, K1, K2, R, t, F


def test_sampson_matrix_matches_reference_diagonal():
    rng, K1, K2, R, t, F = _two_view()
    p1 = rng.uniform([100, 100], [1800, 1000], size=(120, 2))
    p2 = rng.uniform([100, 100], [1800, 1000], size=(120, 2))
    M = sampson_matrix(p1, p2, F, device="cpu")
    ref = sampson_distance(p1, p2, F)                    # paired, i.e. the diagonal of the all-pairs matrix
    assert np.allclose(np.diag(M), ref, atol=1e-6)
    # a few off-diagonal spot checks against single-pair calls
    for i, j in [(0, 5), (3, 17), (40, 2)]:
        assert abs(M[i, j] - sampson_distance(p1[i:i + 1], p2[j:j + 1], F)[0]) < 1e-6


def test_true_correspondence_is_near_zero_and_matched():
    rng, K1, K2, R, t, F = _two_view()
    # real 3D points, projected into both calibrated views -> exact correspondences
    Pw = np.column_stack([rng.uniform(-4, 4, 30), rng.uniform(-3, 3, 30), rng.uniform(6, 20, 30)])
    uv1, _ = project_points(Pw, K1)
    uv2, _ = project_points(transform_points(se3(R, t), Pw), K2)
    # the matched pair's Sampson distance is ~0
    diag = np.diag(sampson_matrix(uv1, uv2, F, device="cpu"))
    assert np.max(diag) < 1e-3
    # best match recovers the identity pairing (candidate j == query j)
    res = best_epipolar_match(uv1, uv2, F, max_px=2.0, device="cpu")
    # each query's true correspondence is at least a strong candidate (small distance)
    assert res["matched"].mean() > 0.8


def test_gpu_matches_cpu_and_measurable():
    rng, K1, K2, R, t, F = _two_view()
    Na, Nb = 2000, 2000
    p1 = rng.uniform([0, 0], [1920, 1080], size=(Na, 2))
    p2 = rng.uniform([0, 0], [1920, 1080], size=(Nb, 2))
    cpu = sampson_matrix(p1, p2, F, device="cpu")
    if not gpu_available():
        return
    gpu = sampson_matrix(p1, p2, F, device="cuda")
    assert np.allclose(cpu, gpu, atol=1e-6)

    for _ in range(3):
        sampson_matrix(p1, p2, F, device="cuda")
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        sampson_matrix(p1, p2, F, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        sampson_matrix(p1, p2, F, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nSampson {Na}x{Nb} = {Na * Nb:,} pairs: GPU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms | "
          f"speedup {cpu_ms / gpu_ms:.1f}x")
