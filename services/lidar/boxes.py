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
                   pitch: float = 0.0, roll: float = 0.0, calib=None) -> dict:
    """Project a cuboid's corners onto a camera image. Returns the 8 corner pixels, which are in front and in
    image, and the edges, so the cuboid can be drawn over the 2D frame (the linked-annotation foundation).
    calib is a resolved Calibration (M-CAL.1); None uses the nominal rig calibration."""
    corners = cuboid_corners(center, dims, yaw, pitch, roll)
    proj = project_to_camera(corners, cam_id, img_w, img_h, calib)
    return {"corners_uv": [[round(float(u), 1), round(float(v), 1)] for u, v in proj["uv"]],
            "in_front": [bool(x) for x in proj["in_front"]], "in_image": [bool(x) for x in proj["in_image"]],
            "edges": BOX_EDGES, "any_in_image": bool(proj["in_image"].any())}


def _bev_rect(center, dims, yaw: float):
    return ((float(center[0]), float(center[1])), (float(dims[0]), float(dims[1])), math.degrees(yaw))


def bev_intersection_area(a: dict, b: dict) -> float:
    """The overlapping footprint area of two oriented boxes via rotated-rectangle intersection."""
    import cv2

    ra = _bev_rect(a["center"], a["dims"], a["yaw"])
    rb = _bev_rect(b["center"], b["dims"], b["yaw"])
    inter_type, region = cv2.rotatedRectangleIntersection(ra, rb)
    if inter_type == 0 or region is None:
        return 0.0
    return float(cv2.contourArea(region))


def iou_bev(a: dict, b: dict) -> float:
    """BEV (top-down) IoU of two oriented boxes."""
    inter = bev_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = a["dims"][0] * a["dims"][1] + b["dims"][0] * b["dims"][1] - inter
    return inter / union if union > 1e-6 else 0.0


def iou_3d(a: dict, b: dict) -> float:
    """True volumetric 3D IoU: the intersection volume (footprint overlap times vertical overlap) over the
    union volume. Not the BEV IoU scaled by a z ratio, which is wrong when the footprints or heights differ."""
    inter_area = bev_intersection_area(a, b)
    if inter_area <= 0:
        return 0.0
    az0, az1 = a["center"][2] - a["dims"][2] / 2, a["center"][2] + a["dims"][2] / 2
    bz0, bz1 = b["center"][2] - b["dims"][2] / 2, b["center"][2] + b["dims"][2] / 2
    z_inter = max(0.0, min(az1, bz1) - max(az0, bz0))
    if z_inter <= 0:
        return 0.0
    inter_vol = inter_area * z_inter
    vol_a = a["dims"][0] * a["dims"][1] * a["dims"][2]
    vol_b = b["dims"][0] * b["dims"][1] * b["dims"][2]
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 1e-6 else 0.0


def cuboid_image_box(center, dims, yaw: float, cam_id: str, img_w: int, img_h: int, calib=None,
                     pitch: float = 0.0, roll: float = 0.0, near_m: float = 0.1,
                     min_px: float = 2.0) -> tuple[list[float] | None, str]:
    """The 2D box a cuboid covers in one camera image, or None and the reason it covers none.

    The box is the image-clipped hull of the cuboid's projection, with every edge first clipped at a near
    plane in front of the lens. Dropping corners behind the camera instead (the hull of the in-front corners
    only) shrinks the box of anything the camera is partly beside, such as a parked car the ego is passing;
    projecting them is worse, since a point behind the lens lands on the wrong side of the image.
    """
    from services.calibration.resolve import nominal_calibration
    from services.lidar.project import _project_cam_points

    cal = calib or nominal_calibration(cam_id, img_w, img_h)
    corners = cuboid_corners(center, dims, yaw, pitch, roll).astype(np.float64)
    cam = (corners - cal.t().astype(np.float64)) @ cal.R().astype(np.float64)
    pts = [cam[i] for i in range(8) if cam[i, 2] > near_m]
    for a, b in BOX_EDGES:
        za, zb = cam[a, 2], cam[b, 2]
        if (za > near_m) != (zb > near_m):
            s = (near_m - za) / (zb - za)
            pts.append(cam[a] + s * (cam[b] - cam[a]))
    if not pts:
        return None, "behind the camera"
    arr = np.asarray(pts, dtype=np.float32)
    proj = _project_cam_points(arr, cal.fx, cal.fy, cal.cx, cal.cy, cal.model, cal.dist, img_w, img_h)
    uv = proj["uv"][proj["in_front"]]
    if len(uv) == 0:
        return None, "behind the camera"
    x0, y0 = float(uv[:, 0].min()), float(uv[:, 1].min())
    x1, y1 = float(uv[:, 0].max()), float(uv[:, 1].max())
    if x1 < 0 or y1 < 0 or x0 >= img_w or y0 >= img_h:
        return None, "outside the image"
    box = [max(0.0, x0), max(0.0, y0), min(float(img_w), x1), min(float(img_h), y1)]
    if box[2] - box[0] < min_px or box[3] - box[1] < min_px:
        return None, f"smaller than {min_px:g} px"
    return [round(v, 1) for v in box], "ok"
