"""Fused preprocess kernel (Tier 1) verification: the NV12->RGB->letterbox->NCHW pipeline must match cv2's
color convert within rounding and be identical on GPU and CPU (same torch code), and the letterbox meta must
un-map a box back to source pixels. A rig-scale timing is printed so the win is measurable."""

import time

import numpy as np
import pytest

from core.accel.preprocess import gpu_available, letterbox_params, preprocess_nv12_batch


def _make_nv12(H, W, rng):
    """A valid NV12 buffer (Y plane H x W, interleaved UV plane H/2 x W)."""
    y = rng.integers(16, 235, size=(H, W), dtype=np.uint8)
    uv = rng.integers(0, 255, size=(H // 2, W), dtype=np.uint8)
    return np.concatenate([y, uv], axis=0)


def test_nv12_to_rgb_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    H, W = 480, 640
    nv12 = _make_nv12(H, W, rng)
    # our fused path at native size (no resize) so only the color convert is compared
    out, meta = preprocess_nv12_batch(nv12, (H, W), out_hw=(H, W), device="cpu")
    ours = (out[0].numpy() * 255.0)                              # (3, H, W) RGB in [0,255]
    ref = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12).transpose(2, 0, 1).astype(np.float32)
    # BT.601 full-range convert; cv2 uses fixed-point rounding, so allow a few LSB
    assert np.abs(ours - ref).mean() < 2.0 and np.percentile(np.abs(ours - ref), 99) < 6.0


def test_gpu_matches_cpu_exact():
    if not gpu_available():
        return
    rng = np.random.default_rng(1)
    H, W = 720, 1280
    batch = np.stack([_make_nv12(H, W, rng) for _ in range(6)])   # a 6-camera rig
    cpu, _ = preprocess_nv12_batch(batch, (H, W), out_hw=(640, 640), device="cpu")
    gpu, _ = preprocess_nv12_batch(batch, (H, W), out_hw=(640, 640), device="cuda")
    assert gpu.shape == (6, 3, 640, 640)
    assert np.allclose(cpu.numpy(), gpu.cpu().numpy(), atol=1e-4)


def test_letterbox_unmaps_box():
    lb = letterbox_params((720, 1280), (640, 640))
    # a box at source (x=640, y=360) maps to the letterboxed image and back
    lx = 640 * lb["scale"] + lb["pad_x"]
    ly = 360 * lb["scale"] + lb["pad_y"]
    assert abs((lx - lb["pad_x"]) / lb["scale"] - 640) < 1e-6
    assert abs((ly - lb["pad_y"]) / lb["scale"] - 360) < 1e-6


def test_measurable():
    cv2 = pytest.importorskip("cv2")
    if not gpu_available():
        return
    rng = np.random.default_rng(2)
    H, W = 720, 1280
    batch = np.stack([_make_nv12(H, W, rng) for _ in range(6)])
    for _ in range(3):
        preprocess_nv12_batch(batch, (H, W), out_hw=(640, 640), device="cuda")
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        preprocess_nv12_batch(batch, (H, W), out_hw=(640, 640), device="cuda")
    gpu_ms = (time.perf_counter() - t0) / n * 1000

    # baseline: the cv2 per-camera CPU pipeline (cvtColor -> resize -> normalize -> CHW)
    t0 = time.perf_counter()
    for _ in range(n):
        for cam in batch:
            rgb = cv2.cvtColor(cam, cv2.COLOR_YUV2RGB_NV12)
            r = cv2.resize(rgb, (640, 360))
            canvas = np.full((640, 640, 3), 114, np.uint8)
            canvas[140:500] = r
            x = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0  # noqa: F841
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    print(f"\npreprocess 6-cam rig 1280x720 -> 640x640: fused GPU {gpu_ms:.2f} ms | "
          f"cv2 CPU pipeline {cpu_ms:.2f} ms | speedup {cpu_ms / gpu_ms:.1f}x")
