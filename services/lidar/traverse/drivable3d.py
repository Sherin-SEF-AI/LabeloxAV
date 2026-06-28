"""Metric 3D drivable surface: lift the 2D drivable concept (M2.2) into a top-down grid of drivable,
non-drivable, and fallback cells. A cell is drivable if it holds road-surface points, non-drivable if an
obstacle rises in it, fallback if it is ground but not classified road (uncertain), and unknown if empty.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud
from services.lidar.traverse.freespace import bev_indices

UNKNOWN, DRIVABLE, NON_DRIVABLE, FALLBACK = 0, 1, 2, 3


def drivable_grid(cloud: Cloud, semantic: np.ndarray | None, road_class_id: int, plane: list[float],
                  res: float | None = None, obstacle_h: float | None = None,
                  x_range: tuple[float, float] = (0.0, 60.0),
                  y_range: tuple[float, float] = (-30.0, 30.0)) -> dict:
    """Grid: 0 unknown, 1 drivable, 2 non-drivable, 3 fallback."""
    cfg = get_settings().lidar
    res = res if res is not None else cfg.traverse_grid_res_m
    obstacle_h = obstacle_h if obstacle_h is not None else cfg.traverse_obstacle_h_m
    gx, gy, nx, ny, valid = bev_indices(cloud.xyz, res, x_range, y_range)
    h = height_above_plane(cloud.xyz, plane)
    flat = gy[valid] * nx + gx[valid]
    hv = h[valid]
    grid = np.zeros(nx * ny, dtype=np.int8)

    observed = np.zeros(nx * ny, dtype=bool)
    observed[flat] = True
    grid[observed] = FALLBACK                              # ground/unclassified by default where observed
    if semantic is not None:
        road = (semantic == road_class_id)[valid]
        grid[flat[road]] = DRIVABLE                        # road-surface cells are drivable
    grid[flat[hv > obstacle_h]] = NON_DRIVABLE             # obstacles override to non-drivable
    grid = grid.reshape(ny, nx)
    n_obs = int(observed.sum())
    return {"grid": grid, "res": res, "x_range": list(x_range), "y_range": list(y_range),
            "drivable_cells": int((grid == DRIVABLE).sum()),
            "non_drivable_cells": int((grid == NON_DRIVABLE).sum()),
            "fallback_cells": int((grid == FALLBACK).sum()),
            "drivable_frac": round(float((grid == DRIVABLE).sum()) / n_obs, 3) if n_obs else 0.0}
