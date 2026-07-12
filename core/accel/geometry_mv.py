"""Multi-view geometric consistency (Tier 2). For each recovered detection in one view, test it against the
candidates in another view by the Sampson epipolar distance, then pick the most consistent match. This is the
all-pairs core of multi-view recall recovery (a candidate manufactured off the primary detector is only trusted
when a second calibrated view agrees on it), and it scales badly on the CPU exactly where the GPU is strongest:
Na queries x Nb candidates Sampson distances per camera pair.

sampson_matrix computes the (Na, Nb) matrix of first-order epipolar distances in one batched pass (matching
core.geometry.sampson_distance per pair), and best_epipolar_match reduces it to the nearest candidate per query
under a pixel gate. Runs on the GPU through torch when a CUDA device is present, else the identical NumPy math,
which is also the correctness oracle."""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_EPS = 1e-12


def gpu_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def _sampson_matrix_np(p1, p2, F):
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
    F = np.asarray(F, dtype=np.float64)
    x1 = np.hstack([p1, np.ones((p1.shape[0], 1))])                  # (Na, 3)
    x2 = np.hstack([p2, np.ones((p2.shape[0], 1))])                  # (Nb, 3)
    Fx1 = x1 @ F.T                                                   # (Na, 3), epipolar line in view 2 per query
    Ftx2 = x2 @ F                                                    # (Nb, 3), epipolar line in view 1 per candidate
    num = Fx1 @ x2.T                                                 # (Na, Nb) = x2 . F x1
    denom = (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2)[:, None] + (Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)[None, :]
    return np.abs(num) / np.sqrt(np.maximum(denom, _EPS))


def _sampson_matrix_torch(p1, p2, F, device):
    f64 = torch.float64
    a = torch.as_tensor(np.asarray(p1, dtype=np.float64).reshape(-1, 2), device=device, dtype=f64)
    b = torch.as_tensor(np.asarray(p2, dtype=np.float64).reshape(-1, 2), device=device, dtype=f64)
    Ft = torch.as_tensor(np.asarray(F, dtype=np.float64), device=device, dtype=f64)
    x1 = torch.cat([a, torch.ones((a.shape[0], 1), device=device, dtype=f64)], dim=1)
    x2 = torch.cat([b, torch.ones((b.shape[0], 1), device=device, dtype=f64)], dim=1)
    Fx1 = x1 @ Ft.T
    Ftx2 = x2 @ Ft
    num = Fx1 @ x2.T
    denom = (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2)[:, None] + (Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)[None, :]
    return torch.abs(num) / torch.sqrt(torch.clamp(denom, min=_EPS))


def sampson_matrix(pts_a, pts_b, F, device=None) -> np.ndarray:
    """All-pairs Sampson epipolar distance in pixels: (Na, Nb) where entry (i, j) is the epipolar distance
    between query a_i (view 1) and candidate b_j (view 2) under the fundamental matrix F. Near zero when the two
    pixels could be the same 3D point seen from the two views."""
    pts_a = np.asarray(pts_a).reshape(-1, 2)
    pts_b = np.asarray(pts_b).reshape(-1, 2)
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return np.zeros((pts_a.shape[0], pts_b.shape[0]), dtype=np.float64)
    if _HAS_TORCH and device != "cpu" and torch.cuda.is_available():
        return _sampson_matrix_torch(pts_a, pts_b, F, torch.device("cuda")).cpu().numpy()
    return _sampson_matrix_np(pts_a, pts_b, F)


def best_epipolar_match(pts_a, pts_b, F, max_px: float = 4.0, device=None) -> dict:
    """For each query in view 1, the nearest candidate in view 2 by Sampson distance, gated at ``max_px``.
    Returns {match_idx (Na,) int (candidate index, -1 if none under the gate), dist_px (Na,), matched (Na,)
    bool}. This is the cross-view association step of multi-view recall recovery."""
    d = sampson_matrix(pts_a, pts_b, F, device=device)
    if d.shape[1] == 0:
        n = d.shape[0]
        return {"match_idx": np.full(n, -1, dtype=np.int64), "dist_px": np.full(n, np.inf),
                "matched": np.zeros(n, dtype=bool)}
    idx = np.argmin(d, axis=1)
    dist = d[np.arange(d.shape[0]), idx]
    matched = dist <= max_px
    return {"match_idx": np.where(matched, idx, -1).astype(np.int64), "dist_px": dist, "matched": matched}
