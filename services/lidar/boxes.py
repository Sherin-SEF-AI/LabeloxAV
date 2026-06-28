"""Oriented 3D box geometry shared across the 3D milestones: the eight corners, ground snapping to the Phase
1 plane, projection of a cuboid onto a camera (M-L1.4), and 3D IoU for tracking and association. The ego
frame is x forward, y left, z up; dims are [L, W, H]; yaw is about ego up, with pitch and roll about the box
x and y axes.
"""

from __future__ import annotations

import math

import numpy as np

from services.lidar.project import project_to_camera

# the 12 edges of a box, as pairs of corner indices (corners ordered by the sign pattern below)
BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def _rot(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def cuboid_corners(center, dims, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The 8 corners of a cuboid in the ego frame, ordered by (x, y, z) sign so BOX_EDGES connects them."""
    length, width, height = dims
    signs = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])
    local = signs * np.array([length, width, height], dtype=np.float64)
    return (np.asarray(center, dtype=np.float64) + local @ _rot(yaw, pitch, roll).T).astype(np.float32)


def ground_z(plane, x: float, y: float) -> float:
    """The ground height at (x, y) for plane ax + by + cz + d = 0."""
    a, b, c, d = plane
    if abs(c) < 1e-6:
        return 0.0
    return -(a * x + b * y + d) / c


def snap_to_ground(center, dims, plane) -> list[float]:
    """Move a cuboid's centre so its bottom face sits on the ground plane at its footprint centre."""
    cx, cy = float(center[0]), float(center[1])
    gz = ground_z(plane, cx, cy)
    return [cx, cy, round(gz + float(dims[2]) / 2.0, 4)]


def project_cuboid(center, dims, yaw: float, cam_id: str, img_w: int, img_h: int,
                   pitch: float = 0.0, roll: float = 0.0) -> dict:
    """Project a cuboid's corners onto a camera image. Returns the 8 corner pixels, which are in front and in
    image, and the edges, so the cuboid can be drawn over the 2D frame (the linked-annotation foundation)."""
    corners = cuboid_corners(center, dims, yaw, pitch, roll)
    proj = project_to_camera(corners, cam_id, img_w, img_h)
    return {"corners_uv": [[round(float(u), 1), round(float(v), 1)] for u, v in proj["uv"]],
            "in_front": [bool(x) for x in proj["in_front"]], "in_image": [bool(x) for x in proj["in_image"]],
            "edges": BOX_EDGES, "any_in_image": bool(proj["in_image"].any())}


def _bev_rect(center, dims, yaw: float):
    return ((float(center[0]), float(center[1])), (float(dims[0]), float(dims[1])), math.degrees(yaw))


def iou_bev(a: dict, b: dict) -> float:
    """BEV (top-down) IoU of two oriented boxes via rotated-rectangle intersection."""
    import cv2

    ra = _bev_rect(a["center"], a["dims"], a["yaw"])
    rb = _bev_rect(b["center"], b["dims"], b["yaw"])
    inter_type, region = cv2.rotatedRectangleIntersection(ra, rb)
    if inter_type == 0 or region is None:
        return 0.0
    inter = float(cv2.contourArea(region))
    area_a = a["dims"][0] * a["dims"][1]
    area_b = b["dims"][0] * b["dims"][1]
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def iou_3d(a: dict, b: dict) -> float:
    """3D IoU: BEV IoU scaled by the vertical overlap of the two boxes."""
    bev = iou_bev(a, b)
    if bev <= 0:
        return 0.0
    az0, az1 = a["center"][2] - a["dims"][2] / 2, a["center"][2] + a["dims"][2] / 2
    bz0, bz1 = b["center"][2] - b["dims"][2] / 2, b["center"][2] + b["dims"][2] / 2
    z_inter = max(0.0, min(az1, bz1) - max(az0, bz0))
    z_union = max(az1, bz1) - min(az0, bz0)
    return bev * (z_inter / z_union) if z_union > 1e-6 else 0.0
