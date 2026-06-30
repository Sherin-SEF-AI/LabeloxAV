"""Milestone F: the unified BEV labeling surface. BEV point rasterization, the metric<->pixel transforms, and
the drivable / freespace metric grids already exist; what was missing is a top-down raster of the LABELS
themselves, so the annotated cuboids, the drivable grid, and the occupancy grid read as one bird's-eye map a
reviewer corrects in metric space. This stamps each labeled cuboid's oriented footprint onto a class grid
(painter's order, later cuboids overwrite on overlap), reusing default_bev_params and metric_to_px. The
footprint test and the rasterizer are pure, so the surface is verified without infra.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from core.logging import get_logger
from services.lidar.bev import default_bev_params, metric_to_px

log = get_logger("bev_labels")

UNLABELED = -1


def in_oriented_box(px: float, py: float, cx: float, cy: float, length: float, width: float, yaw: float) -> bool:
    """Whether metric point (px, py) lies inside an oriented footprint centred at (cx, cy) with the long axis
    (length) along the box x and width along the box y, rotated by yaw radians."""
    dx, dy = px - cx, py - cy
    c, s = math.cos(yaw), math.sin(yaw)
    lx = dx * c + dy * s        # into the box frame
    ly = -dx * s + dy * c
    return abs(lx) <= length / 2 and abs(ly) <= width / 2


def rasterize_label_bev(cuboids: list[dict], params: dict | None = None) -> dict:
    """cuboids: [{center:[x,y,z], dims:[length,width,h], yaw, class_id}]. Returns the grid shape, the count of
    labeled cells, and per-class cell counts. Later cuboids overwrite earlier ones where footprints overlap."""
    p = params or default_bev_params()
    res, w, h = p["res"], p["width"], p["height"]
    grid = np.full((h, w), UNLABELED, dtype=np.int32)
    for cub in cuboids:
        cx, cy = float(cub["center"][0]), float(cub["center"][1])
        length, width = float(cub["dims"][0]), float(cub["dims"][1])
        yaw, cid = float(cub.get("yaw", 0.0)), int(cub["class_id"])
        r = math.hypot(length, width) / 2.0
        steps = int(2 * r / res) + 2
        for i in range(steps):
            mx = cx - r + i * res
            for j in range(steps):
                my = cy - r + j * res
                if not in_oriented_box(mx, my, cx, cy, length, width, yaw):
                    continue
                u, v = metric_to_px(mx, my, p)
                iu, iv = int(u), int(v)
                if 0 <= iv < h and 0 <= iu < w:
                    grid[iv, iu] = cid
    labeled = grid[grid >= 0]
    counts = {int(k): int(n) for k, n in Counter(labeled.tolist()).items()}
    return {"grid_shape": [h, w], "res": res, "labeled_cells": int(labeled.size),
            "class_cell_counts": counts}
