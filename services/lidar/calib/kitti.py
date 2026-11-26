"""KITTI raw calibration, read exactly: the Velodyne -> rectified camera transform as a CameraCalibration.

The KITTI raw importer used to store every drive's camera at rpy=[0, 0, 0], xyz=[1.03, 0, 1.65], a guess that
put the camera on a ground-origin vehicle frame while the clouds it was projected against are in the Velodyne
frame (ground at z = -1.73). Every cuboid projected into a KITTI frame was therefore off by the sensor height
plus the real 1-2 degree mount tilt. The drive's own calibration files say exactly where the camera is:

    x_cam0      = R_velo2cam @ x_velo + T_velo2cam                    (calib_velo_to_cam.txt: R, T)
    x_rect0     = R_rect_00 @ x_cam0                                   (calib_cam_to_cam.txt: R_rect_00)
    [u v 1]^T  ~ P_rect_02 @ [x_rect0; 1] = K2 @ (x_rect0 + b2)        (P_rect_02 = K2 [I | b2])

so the rectified colour camera sees x_cam2 = M @ x_velo + c with M = R_rect_00 @ R_velo2cam and
c = R_rect_00 @ T + b2. This module turns (M, c) into the resolver's parameterisation (services/calibration/
resolve.py: cam = (ego - t) @ R(rpy), R(rpy) = Rz(-yaw) Ry(-pitch) Rx(-roll) R_OPT2EGO), with the Velodyne
frame as the ego frame, because that is the frame KITTI clouds, cuboids and point labels are stored in.

`imu_to_velo` is the lever arm between the OXTS unit (whose pose ego_pose stores) and the Velodyne: placing a
scan in the world from an IMU pose without it moves every point by up to 0.87 m times the heading change.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# optical (x right, y down, z forward) -> ego (x forward, y left, z up); same matrix as services/calibration/resolve
_R_OPT2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def read_calib_file(path: Path) -> dict[str, np.ndarray]:
    """A KITTI `key: v v v` calibration file as float arrays; non-numeric lines (calib_time) are skipped."""
    out: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            continue
    return out


def _zyx_euler(a: np.ndarray) -> tuple[float, float, float]:
    """(z, y, x) angles of a = Rz(z) @ Ry(y) @ Rx(x), in radians."""
    y = -math.asin(max(-1.0, min(1.0, float(a[2, 0]))))
    z = math.atan2(float(a[1, 0]), float(a[0, 0]))
    x = math.atan2(float(a[2, 1]), float(a[2, 2]))
    return z, y, x


def extrinsics_from_affine(m: np.ndarray, c: np.ndarray) -> tuple[list[float], list[float]]:
    """(rpy_deg, xyz_m) such that the resolver's cam = (ego - xyz) @ R(rpy) equals cam = m @ ego + c.

    Row form: cam_row = ego_row @ m.T + c, so R(rpy) = m.T and xyz = -m.T @ c (the camera centre in the ego
    frame). R(rpy) = Rz(-yaw) Ry(-pitch) Rx(-roll) R_OPT2EGO, so the ZYX angles of m.T @ R_OPT2EGO.T are
    (-yaw, -pitch, -roll). m is first projected onto the nearest rotation, as the stored product of two
    rotations carries rounding from the 7-digit calibration files.
    """
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    a = rot.T @ _R_OPT2EGO.T
    z, y, x = _zyx_euler(a)
    rpy = [math.degrees(-x), math.degrees(-y), math.degrees(-z)]
    xyz = (-rot.T @ np.asarray(c, dtype=np.float64)).tolist()
    return [round(v, 6) for v in rpy], [round(v, 6) for v in xyz]


def kitti_camera_calibration(date_dir: Path, cam: str = "02") -> dict:
    """The colour camera's real calibration from a KITTI date folder (calib_cam_to_cam + calib_velo_to_cam).

    Returns the CameraCalibration fields (fx, fy, cx, cy, ref_width, rpy_deg, xyz_m) plus the raw P matrix and
    the Velodyne->camera affine, so a caller can check a projection against KITTI's own formula.
    """
    date_dir = Path(date_dir)
    c2c = read_calib_file(date_dir / "calib_cam_to_cam.txt")
    v2c = read_calib_file(date_dir / "calib_velo_to_cam.txt")
    p = c2c[f"P_rect_{cam}"].reshape(3, 4)
    r_rect = c2c["R_rect_00"].reshape(3, 3)
    r_v2c = v2c["R"].reshape(3, 3)
    t_v2c = v2c["T"].reshape(3)
    k = p[:, :3]
    b = np.linalg.solve(k, p[:, 3])               # P = K [I | b]: the rectified baseline offset of this camera
    m = r_rect @ r_v2c
    c = r_rect @ t_v2c + b
    rpy, xyz = extrinsics_from_affine(m, c)
    size = c2c.get(f"S_rect_{cam}")
    width, height = (int(size[0]), int(size[1])) if size is not None else (1242, 375)
    return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]),
            "ref_width": width, "height": height, "rpy_deg": rpy, "xyz_m": xyz,
            "P": p.tolist(), "velo_to_cam_rect": {"M": m.tolist(), "c": c.tolist()}}


def kitti_imu_to_velo(date_dir: Path) -> dict | None:
    """The OXTS -> Velodyne rigid transform (x_velo = R @ x_imu + T), or None if the file is absent."""
    f = Path(date_dir) / "calib_imu_to_velo.txt"
    if not f.exists():
        return None
    d = read_calib_file(f)
    return {"R": d["R"].reshape(3, 3).round(9).tolist(), "T": d["T"].reshape(3).round(9).tolist()}


def velo_pose_in_imu(imu_to_velo: dict | None) -> np.ndarray:
    """The 4x4 that places Velodyne points in the IMU frame (the inverse of imu_to_velo); identity if unknown."""
    if not imu_to_velo:
        return np.eye(4)
    fwd = np.eye(4)
    fwd[:3, :3] = np.asarray(imu_to_velo["R"], dtype=np.float64)
    fwd[:3, 3] = np.asarray(imu_to_velo["T"], dtype=np.float64)
    # an exact inverse: the stored R is a rotation only to the file's 7 digits, so R.T is not quite its inverse
    return np.linalg.inv(fwd)
