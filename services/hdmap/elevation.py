"""Ground-plane elevation (M3.3): RANSAC plane fit to 3D points, assigning height for the 3D map. The
corpus has no LiDAR, so this fits the GNSS-altitude trajectory where present and reports flat otherwise;
the RANSAC itself is pure numpy and unit-tested on synthetic points, ready for real point clouds."""

from __future__ import annotations

import numpy as np


def ransac_ground_plane(points, iters: int = 200, thresh: float = 0.15, seed: int = 0) -> dict | None:
    """Fit a plane (unit normal, offset d) to points (N,3) by RANSAC, refit on inliers via SVD."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    best, best_in = None, -1
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = float(-n @ p0)
        n_in = int((np.abs(pts @ n + d) < thresh).sum())
        if n_in > best_in:
            best_in, best = n_in, (n, d)
    if best is None:
        return None
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    P = pts[inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[-1]
    d = float(-n @ c)
    return {"normal": n.tolist(), "d": d, "inliers": int(inl.sum()), "total": len(pts)}


def elevation_at(plane: dict, x: float, y: float) -> float | None:
    """Height z on the plane at (x, y): z = -(a x + b y + d) / c."""
    a, b, c = plane["normal"]
    if abs(c) < 1e-9:
        return None
    return -(a * x + b * y + plane["d"]) / c
