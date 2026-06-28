"""3D free space and traversability: a top-down occupancy grid over the cloud. A cell is free if it holds
only near-ground points, occupied if an obstacle rises above it, and unknown if no points fall in it. Reuses
the Phase 1 ground plane to separate ground from obstacles.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def bev_indices(xyz: np.ndarray, res: float, x_range: tuple[float, float],
                y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, int, int]:
    nx = max(1, int((x_range[1] - x_range[0]) / res))
    ny = max(1, int((y_range[1] - y_range[0]) / res))
    gx = ((xyz[:, 0] - x_range[0]) / res).astype(np.int64)
    gy = ((xyz[:, 1] - y_range[0]) / res).astype(np.int64)
    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    return gx, gy, nx, ny, valid


def freespace_grid(cloud: Cloud, plane: list[float], res: float | None = None,
                   obstacle_h: float | None = None, x_range: tuple[float, float] = (0.0, 60.0),
                   y_range: tuple[float, float] = (-30.0, 30.0)) -> dict:
    """An occupancy grid: 0 unknown, 1 free, 2 occupied. Cells with an obstacle above obstacle_h are occupied."""
    cfg = get_settings().lidar
    res = res if res is not None else cfg.traverse_grid_res_m
    obstacle_h = obstacle_h if obstacle_h is not None else cfg.traverse_obstacle_h_m
    gx, gy, nx, ny, valid = bev_indices(cloud.xyz, res, x_range, y_range)
    h = height_above_plane(cloud.xyz, plane)
    flat = (gy[valid] * nx + gx[valid])
    observed = np.zeros(nx * ny, dtype=bool)
    observed[flat] = True
    occ = np.zeros(nx * ny, dtype=bool)
    occ[flat[h[valid] > obstacle_h]] = True
    grid = np.where(occ, OCCUPIED, np.where(observed, FREE, UNKNOWN)).astype(np.int8).reshape(ny, nx)
    n_obs = int(observed.sum())
    return {"grid": grid, "res": res, "x_range": list(x_range), "y_range": list(y_range),
            "free_cells": int((grid == FREE).sum()), "occupied_cells": int((grid == OCCUPIED).sum()),
            "free_frac": round(float((grid == FREE).sum()) / n_obs, 3) if n_obs else 0.0}
