"""Calibration residual field (Tier 3) verification: the batched per-camera residuals must match
core.geometry.reprojection_error, a perfect calibration must give ~0 residual and a perturbed extrinsic must
raise it, and the GPU path must match the CPU path. A rig-scale timing is printed."""

import time

import numpy as np

from core.accel.residual import gpu_available, reprojection_residuals
from core.geometry import project_points, reprojection_error, se3, transform_points


def _rig(rng, C=6):
    from scipy.spatial.transform import Rotation
    Ts, Ks = [], []
    for c in range(C):
        R = Rotation.random(random_state=rng).as_matrix()
        t = rng.uniform(-2, 2, size=3)
        Ts.append(se3(R, t))
        Ks.append(np.array([[900.0 + c, 0, 960], [0, 900.0 + c, 540], [0, 0, 1]]))
    return np.stack(Ts), np.stack(Ks)


def test_matches_geometry_reference():
    rng = np.random.default_rng(0)
    M, C = 300, 6
    Pw = rng.uniform(-6, 6, size=(M, 3))
    Ts, Ks = _rig(rng, C)
    # observed pixels = perfect projection + small noise, so residuals are the noise magnitude
    uv_obs = np.zeros((C, M, 2))
    for c in range(C):
        uv, _ = project_points(transform_points(Ts[c], Pw), Ks[c])
        uv_obs[c] = uv + rng.normal(0, 0.5, size=(M, 2))

    out = reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cpu")
    for c in range(C):
        ref = reprojection_error(Pw, uv_obs[c], Ts[c], Ks[c])
        assert abs(out["per_camera"][c]["rms_px"] - ref["rms_px"]) < 1e-6


def test_perfect_calibration_near_zero_and_perturbation_raises():
    rng = np.random.default_rng(1)
    M = 400
    Pw = rng.uniform(-5, 5, size=(M, 3))
    Ts, Ks = _rig(rng, 1)
    uv, _ = project_points(transform_points(Ts[0], Pw), Ks[0])
    perfect = reprojection_residuals(Pw, uv[None], Ts, Ks, device="cpu")
    assert perfect["per_camera"][0]["rms_px"] < 1e-6            # exact calibration -> zero residual

    bad = Ts.copy()
    bad[0][:3, 3] += 0.1                                        # 10 cm extrinsic error
    perturbed = reprojection_residuals(Pw, uv[None], bad, Ks, device="cpu")
    assert perturbed["per_camera"][0]["rms_px"] > 1.0          # the field lights up


def test_gpu_matches_cpu_and_measurable():
    rng = np.random.default_rng(2)
    M, C = 20000, 6
    Pw = rng.uniform(-10, 10, size=(M, 3))
    Ts, Ks = _rig(rng, C)
    uv_obs = np.zeros((C, M, 2))
    for c in range(C):
        uv, _ = project_points(transform_points(Ts[c], Pw), Ks[c])
        uv_obs[c] = uv
    cpu = reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cpu")
    if not gpu_available():
        return
    gpu = reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cuda")
    # compare where both agree the point is in front (a handful of near-horizon points can flip the z>eps test
    # between the GPU and CPU float reductions; those are excluded from calibration QA anyway)
    assert (cpu["valid"] == gpu["valid"]).mean() > 0.999      # GPU and CPU agree on the in-front mask
    both = cpu["valid"] & gpu["valid"]
    assert np.allclose(cpu["residuals"][both], gpu["residuals"][both], atol=1e-4)

    for _ in range(3):
        reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cuda")
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cuda")
    gpu_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        reprojection_residuals(Pw, uv_obs, Ts, Ks, device="cpu")
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nresidual field {C}x{M} = {C * M:,} residuals: GPU {gpu_ms:.2f} ms | NumPy {cpu_ms:.2f} ms | "
          f"speedup {cpu_ms / gpu_ms:.1f}x")
