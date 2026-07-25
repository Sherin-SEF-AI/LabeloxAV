"""Label propagation / interpolation kernels (the labeling substrate). Propagate sparse keyframe labels to
their neighbors so a human labels few frames and the rest are seeded: warp boxes and masks forward through a
precomputed optical-flow field, batched quaternion SLERP for ego-pose interpolation, and linear box
interpolation between keyframes, all batched over every track. These feed the projection kernel and the
recall-recovery trajectory, and are the large human-effort multiplier of the pipeline.

All run on the GPU through torch when a CUDA device is present, else the identical NumPy math (the oracle).
SLERP matches core.geometry.slerp as a rotation; box interpolation matches recall.interp_box."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-9


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def slerp_batch(q0, q1, t) -> np.ndarray:
    """Batched spherical linear interpolation of (x, y, z, w) quaternions. q0, q1 are (M, 4); t is a scalar or
    (M,) fraction in [0, 1]. Returns (M, 4). Takes the shortest arc (sign-corrected) and falls back to
    normalized lerp for near-parallel quaternions. Represents the same rotation as core.geometry.slerp."""
    a = np.asarray(q0, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(q1, dtype=np.float64).reshape(-1, 4)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size == 1:
        t = np.full(a.shape[0], t[0])
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), _EPS, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), _EPS, None)
    dot = (a * b).sum(axis=1)
    b = np.where(dot[:, None] < 0, -b, b)                # shortest arc
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    near = sin_t < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(near, 1.0, sin_t))
    w1 = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin_t))
    q = w0[:, None] * a + w1[:, None] * b
    return q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), _EPS, None)


def interp_boxes(boxes_a, boxes_b, t) -> np.ndarray:
    """Linear corner interpolation of xyxy boxes, batched. boxes_a, boxes_b (M, 4); t scalar or (M,). frac 0 is
    a, frac 1 is b. Matches recall.interp_box per box."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size == 1:
        t = np.full(a.shape[0], t[0])
    return a + (b - a) * t[:, None]


def _sample_flow_at(flow, xs, ys):
    """Bilinear-sample a (H, W, 2) flow field at pixel coords (xs, ys). NumPy."""
    H, W, _ = flow.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = np.clip(xs - x0, 0, 1)
    wy = np.clip(ys - y0, 0, 1)
    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]
    top = f00 * (1 - wx)[:, None] + f01 * wx[:, None]
    bot = f10 * (1 - wx)[:, None] + f11 * wx[:, None]
    return top * (1 - wy)[:, None] + bot * wy[:, None]


def warp_boxes_by_flow(boxes, flow) -> np.ndarray:
    """Propagate xyxy boxes to the next frame through an optical-flow field (H, W, 2), flow[y, x] = (dx, dy).
    Each corner is displaced by the flow sampled at it. Returns the warped boxes (N, 4) with corners reordered
    so x1 <= x2, y1 <= y2."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    flow = np.asarray(flow, dtype=np.float64)
    tl = _sample_flow_at(flow, b[:, 0], b[:, 1])
    br = _sample_flow_at(flow, b[:, 2], b[:, 3])
    x1, y1 = b[:, 0] + tl[:, 0], b[:, 1] + tl[:, 1]
    x2, y2 = b[:, 2] + br[:, 0], b[:, 3] + br[:, 1]
    return np.stack([np.minimum(x1, x2), np.minimum(y1, y2), np.maximum(x1, x2), np.maximum(y1, y2)], axis=1)


def warp_masks_by_flow(masks, flow, device=None) -> np.ndarray:
    """Propagate (N, H, W) masks to the next frame through an optical-flow field (H, W, 2) via bilinear
    grid_sample: destination pixel p reads the source at p - flow(p). Returns float masks in [0, 1]; threshold
    at 0.5 for a binary mask. Runs on the GPU when available."""
    if not _HAS_TORCH:
        raise RuntimeError("warp_masks_by_flow needs torch")
    m = np.asarray(masks, dtype=np.float32)
    if m.ndim == 2:
        m = m[None]
    n, H, W = m.shape
    fl = np.asarray(flow, dtype=np.float32)
    dev = torch.device("cuda" if (device == "cuda" or (device != "cpu" and torch.cuda.is_available())) else "cpu")
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    flow_t = torch.as_tensor(fl, device=dev)
    src_x = xs - flow_t[..., 0]                           # read the source that flows into this pixel
    src_y = ys - flow_t[..., 1]
    gx = (2.0 * src_x + 1.0) / W - 1.0
    gy = (2.0 * src_y + 1.0) / H - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(n, H, W, 2)
    t = torch.as_tensor(m, device=dev).unsqueeze(1)
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out[:, 0].cpu().numpy()
