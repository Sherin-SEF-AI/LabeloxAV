"""Pseudo-LiDAR: lift the camera-only fleet into 3D point clouds with a metric monocular depth model. This is
the path that makes the module useful on the Tigors today, with no LiDAR hardware.

Per camera: run metric depth, back-project every pixel to a 3D point in the camera frame, rotate into the
ego frame using the rig extrinsics (per-camera yaw + mount height), and fuse all cameras into one cloud.
Optionally place the cloud in a local ENU world frame from GNSS plus heading. The cloud carries its depth
model and calibration version so the lift is reproducible.

Conventions, matching the Phase 3 geo-referencing (services/hdmap/georef.py):
  camera optical frame  x right, y down, z forward
  ego cloud frame       x forward, y left, z up   (KITTI velo convention, what services/lidar/bev.py expects)
The pinhole path back-projects directly; a fisheye surround camera is undistorted to a pinhole virtual
camera first, since a metric depth model trained on rectilinear images is wrong on a raw fisheye frame.

The rig models per-camera yaw and a single mount height (the same model the IPM uses); real 6-DOF extrinsics
override the nominal yaw and the zero x-y mount offset once rig calibration provides them.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_pseudo")

# optical (x right, y down, z forward) -> ego (x forward, y left, z up)
_R_OPT2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)

_DEPTH_MODEL = None
_DEPTH_PROC = None
_DEPTH_ID = None


def _load_depth(model_id: str):
    """Lazy-load the metric depth model once, on the GPU when available."""
    global _DEPTH_MODEL, _DEPTH_PROC, _DEPTH_ID
    if _DEPTH_MODEL is None or _DEPTH_ID != model_id:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _DEPTH_PROC = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _DEPTH_MODEL, _DEPTH_ID = model, model_id
        log.info("lidar.depth_loaded", model=model_id, device=str(next(model.parameters()).device))
    return _DEPTH_MODEL, _DEPTH_PROC


def estimate_depth(image_bgr: np.ndarray, model_id: str | None = None) -> np.ndarray:
    """Metric depth in metres at the image resolution. Depth is distance along the optical axis."""
    import torch

    model_id = model_id or get_settings().lidar.depth_model
    model, proc = _load_depth(model_id)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt").to(model.device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=image_bgr.shape[:2], mode="bicubic", align_corners=False).squeeze(1).squeeze(0)
    return depth.float().cpu().numpy()


def _undistort_fisheye(image_bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                       dist: list[float]) -> np.ndarray:
    """Rectify a fisheye frame to a pinhole virtual camera with the same intrinsics, so the metric depth
    model and the pinhole back-projection are both valid."""
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.array((list(dist) + [0.0, 0.0, 0.0, 0.0])[:4], dtype=np.float64).reshape(4, 1)
    return cv2.fisheye.undistortImage(image_bgr, k, d, Knew=k)


def backproject(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                luma: np.ndarray, stride: int = 2, z_min: float = 0.5,
                z_max: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole back-projection of a metric depth map to camera-frame xyz, with per-point intensity from the
    image luma. Strided for density and clipped to a sane metric range."""
    h, w = depth_m.shape
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    z = z[valid]
    uu, vv = uu[valid], vv[valid]
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    inten = luma[vv, uu].astype(np.float32)
    return xyz, inten


def camera_to_ego(xyz_cam: np.ndarray, cam_yaw_deg: float, height_m: float) -> np.ndarray:
    """Rotate camera-frame points into the ego frame: optical axes to (forward, left, up), then the camera
    mounting yaw about ego up, then lift by the mount height. The yaw sign matches the Phase 3 compass
    convention (cam_l=-90 puts left-camera-forward to the vehicle's left)."""
    ego = xyz_cam @ _R_OPT2EGO.T
    theta = -math.radians(cam_yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ego = ego @ rz.T
    ego[:, 2] += height_m
    return ego


def lift_frame_group(images: dict[str, np.ndarray], ts_ns: int, calibration_version: str,
                     model_id: str | None = None, stride: int | None = None,
                     max_points: int | None = None) -> Cloud:
    """Fuse a synchronized multi-camera frame group into one ego-frame pseudo-LiDAR cloud."""
    cfg = get_settings()
    model_id = model_id or cfg.lidar.depth_model
    stride = stride if stride is not None else 2
    max_points = max_points if max_points is not None else cfg.lidar.viewer_max_points
    height = cfg.spatial.camera_height_m

    xyz_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    for cam_id, image_bgr in images.items():
        lens_name = cfg.rig.camera_lens.get(cam_id, "narrow")
        k = cfg.rig.lenses[lens_name]
        h, w = image_bgr.shape[:2]
        scale = w / cfg.rig.ref_width
        fx, fy, cx, cy = k.fx * scale, k.fy * scale, w / 2.0, h / 2.0
        work = _undistort_fisheye(image_bgr, fx, fy, cx, cy, k.dist) if k.model == "fisheye" else image_bgr
        depth = estimate_depth(work, model_id)
        luma = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        xyz_cam, inten = backproject(depth, fx, fy, cx, cy, luma, stride=stride)
        ego = camera_to_ego(xyz_cam, cfg.rig.camera_yaw_deg.get(cam_id, 0.0), height)
        xyz_parts.append(ego)
        int_parts.append(inten)
        log.info("lidar.pseudo_cam", cam=cam_id, points=int(ego.shape[0]),
                 depth_range=[round(float(depth.min()), 2), round(float(depth.max()), 2)])

    xyz = np.vstack(xyz_parts) if xyz_parts else np.zeros((0, 3), np.float32)
    inten = np.concatenate(int_parts) if int_parts else np.zeros((0,), np.float32)
    cloud = Cloud(xyz=xyz, intensity=inten, ts_ns=ts_ns, source="pseudo", frame="ego",
                  depth_model=model_id, calibration_version=calibration_version)
    if cloud.n > max_points:
        cloud = cloud.decimate(max_points, seed=ts_ns % (2**31))
    return cloud


def place_in_world(cloud: Cloud, lat: float, lon: float, heading_rad: float) -> Cloud:
    """Place an ego-frame cloud into a local ENU world frame (x east, y north, z up) anchored at the GNSS
    position, using the heading (radians from north, clockwise). The lateral sign follows the Phase 3
    geo-referencing (lateral is +right of the vehicle axis), so this composes with vehicle_to_world."""
    fwd = cloud.xyz[:, 0]
    lateral_right = -cloud.xyz[:, 1]   # ego y is +left; the world transform wants +right
    up = cloud.xyz[:, 2]
    s, c = math.sin(heading_rad), math.cos(heading_rad)
    east = fwd * s + lateral_right * c
    north = fwd * c - lateral_right * s
    world = np.stack([east, north, up], axis=1).astype(np.float32)
    return Cloud(xyz=world, intensity=cloud.intensity, ts_ns=cloud.ts_ns, ring=cloud.ring,
                 source=cloud.source, frame="world_enu", depth_model=cloud.depth_model,
                 calibration_version=cloud.calibration_version)
