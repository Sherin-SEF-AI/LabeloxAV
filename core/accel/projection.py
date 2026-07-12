"""Fused projection kernel (Tier 1). One pass, batched over all boxes x all cameras in a session:

    world point -> SE(3) world->camera -> perspective divide -> distortion -> intrinsics -> bounds check

The per-frame ego-pose interpolation (SLERP) that produces each camera's world->camera pose is composed by the
caller (extrinsic o interpolated ego pose) and passed in as ``T_cam_world``; this kernel is the heavy inner
loop that runs per annotation, per camera, per frame, so a session-scale batch turns a NumPy-bound Python loop
into one vectorized GPU pass.

Two distortion models: ``pinhole`` (Brown-Conrady k1,k2,p1,p2,k3, matching core.geometry.project_points) and
``fisheye`` (Kannala-Brandt equidistant k1..k4, for the ISX031 wide rigs). The kernel runs on the GPU through
torch when a CUDA device is present, and falls back to the identical NumPy math otherwise; the NumPy path is
also the correctness oracle the tests check the GPU path against, so verification is free.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # torch is an optional heavy dep; the NumPy path always works
    _HAS_TORCH = False

_EPS = 1e-9


def gpu_available() -> bool:
    """True when a CUDA device is present and the GPU path will be taken by default."""
    return _HAS_TORCH and torch.cuda.is_available()


# --------------------------------------------------------------------------- NumPy backend

def _distort_pinhole_np(x, y, dist):
    # x, y are (C, M); dist is (C, D); per-camera coeffs are (C, 1) so they broadcast over M
    k1, k2, p1, p2 = dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4]
    k3 = dist[:, 4:5] if dist.shape[1] > 4 else 0.0
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _distort_fisheye_np(x, y, dist):
    k1, k2, k3, k4 = dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 * t2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    scale = np.where(r > _EPS, theta_d / np.where(r > _EPS, r, 1.0), 1.0)
    return x * scale, y * scale


def _project_world_np(points_world, T_cam_world, K, dist, img_wh, model):
    pts = np.asarray(points_world, dtype=np.float64)                      # (M, 3)
    T = np.asarray(T_cam_world, dtype=np.float64)                         # (C, 4, 4)
    K = np.asarray(K, dtype=np.float64)                                   # (C, 3, 3)
    R = T[:, :3, :3]                                                      # (C, 3, 3)
    t = T[:, :3, 3]                                                       # (C, 3)
    p_cam = np.einsum("cij,mj->cmi", R, pts) + t[:, None, :]             # (C, M, 3)
    z = p_cam[..., 2]
    valid = z > _EPS
    zc = np.where(valid, z, 1.0)
    x = p_cam[..., 0] / zc
    y = p_cam[..., 1] / zc
    if dist is not None:
        d = np.asarray(dist, dtype=np.float64).reshape(len(K), -1)       # (C, D)
        x, y = (_distort_fisheye_np if model == "fisheye" else _distort_pinhole_np)(x, y, d)
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]
    u = fx * x + cx
    v = fy * y + cy
    uv = np.stack([u, v], axis=-1)                                       # (C, M, 2)
    in_bounds = valid
    if img_wh is not None:
        wh = np.asarray(img_wh, dtype=np.float64)                        # (C, 2)
        w = wh[:, 0][:, None]
        h = wh[:, 1][:, None]
        in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv, valid, in_bounds


# --------------------------------------------------------------------------- torch backend

def _distort_pinhole_t(x, y, dist):
    # x, y are (C, M); dist is (C, D); per-camera coeffs (C, 1) broadcast over M
    k1, k2, p1, p2 = dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4]
    k3 = dist[:, 4:5] if dist.shape[1] > 4 else torch.zeros_like(k1)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _distort_fisheye_t(x, y, dist):
    k1, k2, k3, k4 = dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4]
    r = torch.sqrt(x * x + y * y)
    theta = torch.atan(r)
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 * t2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    safe_r = torch.where(r > _EPS, r, torch.ones_like(r))
    scale = torch.where(r > _EPS, theta_d / safe_r, torch.ones_like(r))
    return x * scale, y * scale


def _project_world_torch(points_world, T_cam_world, K, dist, img_wh, model, device):
    f64 = torch.float64
    pts = torch.as_tensor(np.asarray(points_world, dtype=np.float64), device=device, dtype=f64)     # (M,3)
    T = torch.as_tensor(np.asarray(T_cam_world, dtype=np.float64), device=device, dtype=f64)        # (C,4,4)
    Kt = torch.as_tensor(np.asarray(K, dtype=np.float64), device=device, dtype=f64)                 # (C,3,3)
    R = T[:, :3, :3]
    tvec = T[:, :3, 3]
    p_cam = torch.einsum("cij,mj->cmi", R, pts) + tvec[:, None, :]                                  # (C,M,3)
    z = p_cam[..., 2]
    valid = z > _EPS
    zc = torch.where(valid, z, torch.ones_like(z))
    x = p_cam[..., 0] / zc
    y = p_cam[..., 1] / zc
    if dist is not None:
        d = torch.as_tensor(np.asarray(dist, dtype=np.float64).reshape(Kt.shape[0], -1),
                            device=device, dtype=f64)                    # (C, D)
        x, y = (_distort_fisheye_t if model == "fisheye" else _distort_pinhole_t)(x, y, d)
    fx, fy = Kt[:, 0, 0][:, None], Kt[:, 1, 1][:, None]
    cx, cy = Kt[:, 0, 2][:, None], Kt[:, 1, 2][:, None]
    u = fx * x + cx
    v = fy * y + cy
    uv = torch.stack([u, v], dim=-1)
    in_bounds = valid
    if img_wh is not None:
        wh = torch.as_tensor(np.asarray(img_wh, dtype=np.float64), device=device, dtype=f64)
        w, h = wh[:, 0][:, None], wh[:, 1][:, None]
        in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv.cpu().numpy(), valid.cpu().numpy(), in_bounds.cpu().numpy()


# --------------------------------------------------------------------------- public API

def project_world_batch(points_world, T_cam_world, K, dist=None, img_wh=None, model="pinhole", device=None):
    """Project M world points into C cameras in one batched pass.

    Shapes: points_world (M, 3); T_cam_world (C, 4, 4) world->camera per camera; K (C, 3, 3);
    dist (C, D) distortion per camera (D=5 pinhole, D=4 fisheye) or None; img_wh (C, 2) optional for the bounds
    check. Returns (uv (C, M, 2), valid (C, M) z>0, in_bounds (C, M)). ``device`` forces "cpu"/"cuda"; by
    default the GPU is used when available. ``model`` is "pinhole" (Brown-Conrady) or "fisheye" (Kannala-
    Brandt), matching core.geometry.project_points and cv2.fisheye respectively."""
    if model not in ("pinhole", "fisheye"):
        raise ValueError(f"unknown model {model!r}")
    T_cam_world = np.atleast_3d(np.asarray(T_cam_world, dtype=np.float64)).reshape(-1, 4, 4)
    K = np.asarray(K, dtype=np.float64).reshape(-1, 3, 3)
    if _HAS_TORCH and device != "cpu" and (device == "cuda" or torch.cuda.is_available()):
        dev = torch.device("cuda" if (device == "cuda" or torch.cuda.is_available()) else "cpu")
        return _project_world_torch(points_world, T_cam_world, K, dist, img_wh, model, dev)
    return _project_world_np(points_world, T_cam_world, K, dist, img_wh, model)
