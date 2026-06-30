"""2D-to-3D lifting: the primary, robust 3D-box source for the camera fleet. For a 2D object (box, optional
mask, class, track) on a frame with a synchronized cloud, form a frustum with the Phase 1 projection, gather
the points inside it, isolate the object surface by depth, fit an oriented cuboid in the BEV, and snap it to
the Phase 1 ground plane. The cuboid inherits the 2D class and track, so it joins the same identity.

Reuses services/lidar/project (M-L1.4) and the ground plane from services/lidar/clean/ground (M-L1.2). The
ego frame is x forward, y left, z up; yaw is about ego up, 0 = facing forward.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.project import project_to_camera

log = get_logger("lidar_lift")


def frustum_indices(cloud_xyz: np.ndarray, bbox: list[float], cam_id: str,
                    img_w: int, img_h: int, pad: float = 0.0) -> np.ndarray:
    """Indices of cloud points that project inside the 2D box and lie in front of the camera."""
    proj = project_to_camera(cloud_xyz, cam_id, img_w, img_h)
    uv, in_front = proj["uv"], proj["in_front"]
    x1, y1, x2, y2 = bbox
    if pad:
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    inside = in_front & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    return np.nonzero(inside)[0]


def _isolate_surface(points: np.ndarray, depth_gate: float) -> np.ndarray:
    """A frustum sees the object and the background behind it. Keep the nearest coherent depth band, which is
    the object surface, dropping the far background that shares the box."""
    if len(points) == 0:
        return points
    rng = np.linalg.norm(points[:, :2], axis=1)   # horizontal range in the ego frame
    near = np.percentile(rng, 10)
    return points[rng <= near + depth_gate]


def _plane_z(plane: list[float], x: float, y: float) -> float:
    """The ground height at (x, y) for plane ax + by + cz + d = 0."""
    a, b, c, d = plane
    if abs(c) < 1e-6:
        return 0.0
    return -(a * x + b * y + d) / c


def fit_cuboid(points_ego: np.ndarray, ground_plane: list[float] | None = None,
               depth_gate: float | None = None, min_points: int | None = None) -> dict | None:
    """Fit an oriented cuboid to frustum points: a minimum-area rectangle in the BEV gives the centre, L, W
    and yaw; the height comes from the z extent; the bottom snaps to the ground plane. None if too sparse."""
    cfg = get_settings().lidar
    depth_gate = depth_gate if depth_gate is not None else cfg.lift_depth_gate_m
    min_points = min_points if min_points is not None else cfg.lift_min_frustum_points

    pts = _isolate_surface(np.asarray(points_ego, dtype=np.float32), depth_gate)
    if len(pts) < min_points:
        return None

    # exclude the road surface so the fitted footprint is the object, not the ground sharing its frustum
    fit_pts = pts
    if ground_plane is not None:
        a, b, c, d = ground_plane
        if abs(c) > 1e-6:
            above = pts[:, 2] - (-(a * pts[:, 0] + b * pts[:, 1] + d) / c)
            non_ground = pts[above > 0.2]
            if len(non_ground) >= min_points:
                fit_pts = non_ground

    xy = fit_pts[:, :2].astype(np.float32)
    (cx, cy), (w0, h0), angle_deg = cv2.minAreaRect(xy)
    length, width = (max(w0, h0), min(w0, h0))
    # yaw aligns to the longer side; minAreaRect angle is for the (w0, h0) order
    yaw = math.radians(angle_deg + (90.0 if h0 > w0 else 0.0))
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))   # wrap to (-pi, pi]

    z_top = float(np.percentile(fit_pts[:, 2], 97))
    ground_z = _plane_z(ground_plane, cx, cy) if ground_plane else float(np.percentile(pts[:, 2], 3))
    height = max(z_top - ground_z, 0.3)
    center_z = ground_z + height / 2.0

    # fill is how densely the points occupy the fitted footprint, a fit-quality signal for the gate
    footprint = max(length * width, 1e-3)
    fill = min(1.0, len(fit_pts) / (footprint * 80.0))
    return {"center": [round(float(cx), 3), round(float(cy), 3), round(center_z, 3)],
            "dims": [round(float(length), 3), round(float(width), 3), round(float(height), 3)],
            "yaw": round(float(yaw), 4), "pitch": 0.0, "roll": 0.0,
            "n_points": int(len(fit_pts)), "fill": round(float(fill), 3),
            "ground_z": round(float(ground_z), 3)}


def lift_box(cloud_xyz: np.ndarray, bbox: list[float], cam_id: str, img_w: int, img_h: int,
             ground_plane: list[float] | None = None, mask_indices: np.ndarray | None = None) -> dict | None:
    """Lift one 2D box to an oriented, ground-snapped cuboid. mask_indices (cloud points inside the 2D mask)
    tighten the frustum when available; otherwise the box frustum is used."""
    idx = frustum_indices(cloud_xyz, bbox, cam_id, img_w, img_h)
    if mask_indices is not None and len(mask_indices):
        idx = np.intersect1d(idx, mask_indices, assume_unique=False)
    if len(idx) == 0:
        return None
    cuboid = fit_cuboid(cloud_xyz[idx], ground_plane)
    if cuboid is not None:
        cuboid["box_source"] = "lifted"
    return cuboid
