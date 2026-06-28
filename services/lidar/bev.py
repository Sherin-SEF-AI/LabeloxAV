"""LiDAR bird's-eye-view (BEV) annotation support.

Real point-cloud cuboid labeling without a 3D engine: rasterize the cloud top-down into an image the
existing editor renders, annotate oriented boxes on it (the rotated-box tool), then lift each box back to
a metric 3D cuboid using the BEV projection and the points the box encloses (for the z extent). This is
the standard production workflow for LiDAR cuboids; a full 3D viewer is an optional refinement on top.

Frame convention (KITTI Velodyne): x forward, y left, z up, metres. BEV image: forward is up (row 0 is the
far range), the ego is at the bottom centre, and the left of the scene is the left of the image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def default_bev_params(x_max: float = 70.4, y_abs: float = 40.0, res: float = 0.1) -> dict:
    """A standard forward-looking KITTI BEV window: x in [0, x_max], y in [-y_abs, y_abs], res m/pixel."""
    return {
        "x_min": 0.0, "x_max": float(x_max), "y_min": -float(y_abs), "y_max": float(y_abs),
        "res": float(res),
        "width": int(round(2 * y_abs / res)),
        "height": int(round(x_max / res)),
    }


def metric_to_px(x: float, y: float, p: dict) -> tuple[float, float]:
    """Vehicle (x forward, y left) metres -> BEV pixel (u col, v row)."""
    u = (p["y_max"] - y) / p["res"]
    v = (p["x_max"] - x) / p["res"]
    return u, v


def px_to_metric(u: float, v: float, p: dict) -> tuple[float, float]:
    """BEV pixel (u col, v row) -> vehicle (x forward, y left) metres."""
    y = p["y_max"] - u * p["res"]
    x = p["x_max"] - v * p["res"]
    return x, y


def rasterize_bev(points: np.ndarray, p: dict) -> np.ndarray:
    """points: N x >=3 (x, y, z[, intensity]). Returns a BGR BEV image coloured by height, with the highest
    point per pixel kept so objects sit above the ground plane."""
    w, h = p["width"], p["height"]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    m = (x >= p["x_min"]) & (x < p["x_max"]) & (y >= p["y_min"]) & (y < p["y_max"])
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return img
    u = np.clip(((p["y_max"] - y) / p["res"]).astype(np.int32), 0, w - 1)
    v = np.clip(((p["x_max"] - x) / p["res"]).astype(np.int32), 0, h - 1)
    order = np.argsort(z)  # draw low points first so higher points overwrite (objects above ground)
    cval = (np.clip((z[order] + 3.0) / 6.0, 0.0, 1.0) * 255).astype(np.uint8)  # height in [-3, 3] m
    colors = cv2.applyColorMap(cval.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    img[v[order], u[order]] = colors
    return cv2.dilate(img, np.ones((2, 2), np.uint8))  # thicken sparse returns so they are visible


def bev_box_to_cuboid(bbox: list[float], rot_deg: float, points: np.ndarray, p: dict,
                      default_h: float = 1.6, ground_z: float = -1.7) -> dict:
    """Lift an oriented BEV box (pixel AABB [u0,v0,u1,v1] + rotation in degrees) to an ego-frame cuboid
    {center:[x,y,z], size:[w,l,h], yaw}. The footprint comes from the BEV projection; z and height come from
    the points the box encloses (data-driven), with a default when too few points fall inside."""
    u0, v0, u1, v1 = bbox
    uc, vc = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    w = abs(u1 - u0) * p["res"]   # lateral extent (width)
    length = abs(v1 - v0) * p["res"]  # forward extent (length)
    cx, cy = px_to_metric(uc, vc, p)
    yaw = -math.radians(rot_deg)  # image clockwise -> vehicle CCW about +z (image y is flipped vs +y left)

    # enclosed points: rotate the cloud into the box frame and test the rectangle
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    dx, dy = x - cx, y - cy
    c, s = math.cos(-yaw), math.sin(-yaw)
    along_l = c * dx - s * dy
    along_w = s * dx + c * dy
    inside = (np.abs(along_l) <= length / 2.0) & (np.abs(along_w) <= w / 2.0)
    n = int(inside.sum())
    if n >= 5:
        zin = z[inside]
        z_min, z_max = float(zin.min()), float(zin.max())
        height = max(0.5, z_max - z_min)
        cz = (z_min + z_max) / 2.0
    else:
        height = default_h
        cz = ground_z + height / 2.0

    return {
        "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "size": [round(w, 3), round(length, 3), round(height, 3)],
        "yaw": round(yaw, 4),
        "n_points": n,
    }
