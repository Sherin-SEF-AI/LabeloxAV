"""Batched image-QA metrics (Tier 3) verification: the Laplacian-variance sharpness must match
cv2.Laplacian(...).var() per frame, exposure/clipping must match the plain NumPy stats, GPU must match CPU, and
the rig exposure-consistency spread must flag an inconsistent rig. A session-scale timing is printed."""

import time

import numpy as np
import pytest

from core.accel.imgqa import gpu_available, image_quality_batch, rig_exposure_consistency


def test_sharpness_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    N, H, W = 12, 240, 320
    # a mix of sharp (noise) and blurred frames
    frames = rng.integers(0, 255, size=(N, H, W), dtype=np.uint8)
    for i in range(0, N, 2):
        frames[i] = cv2.GaussianBlur(frames[i], (7, 7), 3)

    out = image_quality_batch(frames, device="cpu")
    for i in range(N):
        ref = float(cv2.Laplacian(frames[i].astype(np.float64), cv2.CV_64F).var())
        # float32 compute vs cv2's float64 -> agree to a few parts in 1e4, far tighter than any QA threshold
        assert abs(out["blur"][i] - ref) / max(ref, 1.0) < 1e-3, f"frame {i} sharpness mismatch"
    # exposure + clipping match plain numpy
    assert np.allclose(out["mean_luma"], frames.reshape(N, -1).mean(1), atol=1e-6)
    assert np.allclose(out["clipped_hi"], (frames >= 250).reshape(N, -1).mean(1), atol=1e-9)


def test_gpu_matches_cpu():
    if not gpu_available():
        return
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, size=(24, 360, 640), dtype=np.uint8)
    cpu = image_quality_batch(frames, device="cpu")
    gpu = image_quality_batch(frames, device="cuda")
    assert np.allclose(cpu["blur"], gpu["blur"], rtol=1e-6)
    assert np.allclose(cpu["mean_luma"], gpu["mean_luma"], atol=1e-6)


def test_rig_exposure_consistency_flags_spread():
    even = rig_exposure_consistency([120, 122, 118, 121, 119, 120])
    uneven = rig_exposure_consistency([120, 30, 118, 240, 119, 120])   # one dark, one washed out
    assert even["std"] < 2.0
    assert uneven["std"] > 40.0 and uneven["range"] > 200.0


def test_measurable():
    cv2 = pytest.importorskip("cv2")
    if not gpu_available():
        return
    rng = np.random.default_rng(2)
    N, H, W = 120, 720, 1280                              # a session's worth of frames
    frames = rng.integers(0, 255, size=(N, H, W), dtype=np.uint8)
    for _ in range(3):
        image_quality_batch(frames, device="cuda")
    t0 = time.perf_counter()
    image_quality_batch(frames, device="cuda")
    gpu_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for f in frames:
        _ = cv2.Laplacian(f.astype(np.float64), cv2.CV_64F).var()
        _ = f.mean()
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"\nimg QA {N} frames {H}x{W}: fused GPU {gpu_ms:.2f} ms | cv2 per-frame loop {cpu_ms:.2f} ms | "
          f"speedup {cpu_ms / gpu_ms:.1f}x")
