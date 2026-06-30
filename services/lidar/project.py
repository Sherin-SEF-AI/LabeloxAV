"""Project cloud points into the synchronized camera frames and back, using the Phase 3 calibration. This is
the inverse of the pseudo-LiDAR lift (services/lidar/ingest/pseudo.py) and the foundation for the 3D-to-2D
linked annotation in Phase 2: a point in the cloud maps to a pixel, and a pixel maps to a ray (and, with the
cloud, to the nearest 3D point along it).

Ego frame is x forward, y left, z up; camera optical frame is x right, y down, z forward. The pinhole path
projects directly; a fisheye surround camera distorts through the equidistant model so the pixel matches the
raw frame. Accuracy is gated on calibration (M-L1.5): a session that fails validation is excluded.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.pseudo import _R_OPT2EGO

log = get_logger("lidar_project")


def _intrinsics(cam_id: str, img_w: int, img_h: int):
    cfg = get_settings()
    lens_name = cfg.rig.camera_lens.get(cam_id, "narrow")
    k = cfg.rig.lenses[lens_name]
    scale = img_w / cfg.rig.ref_width
    return k, k.fx * scale, k.fy * scale, img_w / 2.0, img_h / 2.0


def _ego_to_camera_matrix(cam_id: str) -> tuple[np.ndarray, float]:
    """The 3x3 that maps ego xyz (minus the mount height) to the camera optical frame, plus the height."""
    cfg = get_settings()
    yaw = cfg.rig.camera_yaw_deg.get(cam_id, 0.0)
    theta = -math.radians(yaw)
    c, s = math.cos(theta), math.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    # forward lift was ego = (cam @ R_opt2ego.T) @ rz.T + t, so cam = (ego - t) @ (rz @ R_opt2ego)
    m = rz @ _R_OPT2EGO
    return m.astype(np.float32), cfg.spatial.camera_height_m


def ego_to_camera(ego_xyz: np.ndarray, cam_id: str) -> np.ndarray:
    """Ego-frame points to the camera optical frame (x right, y down, z forward)."""
    m, height = _ego_to_camera_matrix(cam_id)
    shifted = ego_xyz.astype(np.float32).copy()
    shifted[:, 2] -= height
    return shifted @ m


def _project_cam_points(cam: np.ndarray, fx: float, fy: float, cx: float, cy: float, model: str,
                        dist, img_w: int, img_h: int) -> dict:
    """Shared pinhole/fisheye projection of camera-optical-frame points to pixels."""
    z = cam[:, 2]
    in_front = z > 1e-3
    uv = np.full((cam.shape[0], 2), -1.0, dtype=np.float32)
    if model == "fisheye" and np.any(in_front):
        km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        d = np.array((list(dist) + [0, 0, 0, 0])[:4], dtype=np.float64).reshape(4, 1)
        pts = cam[in_front].astype(np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), km, d)
        uv[in_front] = proj.reshape(-1, 2).astype(np.float32)
    else:
        safe_z = np.where(in_front, z, 1.0)
        uv[:, 0] = cam[:, 0] / safe_z * fx + cx
        uv[:, 1] = cam[:, 1] / safe_z * fy + cy
    in_image = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    return {"uv": uv, "in_front": in_front, "in_image": in_image, "depth": z}


def project_to_camera(ego_xyz: np.ndarray, cam_id: str, img_w: int, img_h: int, calib=None) -> dict:
    """Project ego-frame points to pixels. Returns uv, the in-front and in-image masks, and the optical depth.
    calib is a resolved Calibration (M-CAL.1); when None the nominal rig calibration is used, reproducing the
    legacy projection exactly. Pass a session-resolved calib to project through real calibration."""
    if calib is None:
        k, fx, fy, cx, cy = _intrinsics(cam_id, img_w, img_h)
        cam = ego_to_camera(ego_xyz, cam_id)
        model, dist = k.model, k.dist
    else:
        fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
        cam = (ego_xyz.astype(np.float32) - calib.t()) @ calib.R()
        model, dist = calib.model, calib.dist
    return _project_cam_points(cam, fx, fy, cx, cy, model, dist, img_w, img_h)


def camera_ray_to_ego(u: float, v: float, cam_id: str, img_w: int, img_h: int) -> dict:
    """A pixel to a ray in the ego frame: the camera mount origin and a unit direction. With a cloud, the
    nearest point along this ray is the 3D point the pixel sees (lift_pixel)."""
    k, fx, fy, cx, cy = _intrinsics(cam_id, img_w, img_h)
    if k.model == "fisheye":
        km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        d = np.array((list(k.dist) + [0, 0, 0, 0])[:4], dtype=np.float64).reshape(4, 1)
        und = cv2.fisheye.undistortPoints(np.array([[[float(u), float(v)]]], dtype=np.float64), km, d)
        dir_cam = np.array([und[0, 0, 0], und[0, 0, 1], 1.0], dtype=np.float32)
    else:
        dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
    m, height = _ego_to_camera_matrix(cam_id)
    dir_ego = m @ dir_cam                      # cam = ego_shifted @ m  ->  ego_dir = m @ cam_dir
    dir_ego = dir_ego / (np.linalg.norm(dir_ego) or 1.0)
    origin = np.array([0.0, 0.0, height], dtype=np.float32)
    return {"origin": origin, "direction": dir_ego.astype(np.float32)}


def lift_pixel(u: float, v: float, cam_id: str, cloud: Cloud, img_w: int, img_h: int,
               max_dist: float = 0.5) -> dict | None:
    """The 3D cloud point a pixel sees: the closest point to the camera ray within max_dist. The seam for
    Phase 2 click-to-3D linked annotation."""
    ray = camera_ray_to_ego(u, v, cam_id, img_w, img_h)
    rel = cloud.xyz - ray["origin"]
    t = rel @ ray["direction"]
    in_front = t > 0
    if not np.any(in_front):
        return None
    closest = ray["origin"] + np.outer(t, ray["direction"])
    perp = np.linalg.norm(cloud.xyz - closest, axis=1)
    perp[~in_front] = np.inf
    idx = int(np.argmin(perp))
    if perp[idx] > max_dist:
        return None
    return {"index": idx, "point": [float(x) for x in cloud.xyz[idx]], "range": float(t[idx]),
            "ray_offset": float(perp[idx])}
