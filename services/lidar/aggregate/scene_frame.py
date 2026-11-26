"""The session scene frame: every scan of a session placed in one world frame by its ego pose.

`ego_pose` (0112) says where the vehicle was at each instant in a session-local ENU frame. A scan's points are
in the sensor's ego frame at that instant. So T_world_ego(scan) = T_world_pose(ts) @ T_pose_ego, where
T_pose_ego is the lever arm between the unit that measured the pose and the sensor the cloud is in: identity
on the camera fleet (the pose is the ego's), the OXTS -> Velodyne transform on KITTI (session.sensors
.imu_to_velo, written by scripts/import_kitti_raw.py). Ignoring that lever arm on KITTI moves a static cuboid by
0.87 m times the heading change, which over a turn is a whole lane.

The scene frame is what "label once in 3D" draws in: accumulate the scans into one static cloud, draw a cuboid
there, and the inverse of each scan's T_world_ego puts that cuboid on every scan.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("scene_frame")

# A pose this far from a cloud's timestamp is still the pose of that scan (KITTI's OXTS and Velodyne are
# stamped a few milliseconds apart); further than this it is a different instant.
POSE_MATCH_NS = 60_000_000


@dataclass
class ScenePose:
    cloud_id: uuid.UUID
    ts_ns: int
    cloud_uri: str
    T_world_ego: np.ndarray      # 4x4, ego (sensor) frame -> session world frame
    pose_ts_ns: int


def quat_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """3x3 rotation of a unit quaternion."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_matrix(pose) -> np.ndarray:
    """4x4 of an EgoPose row (or anything with x, y, z, qw, qx, qy, qz)."""
    t = np.eye(4)
    t[:3, :3] = quat_matrix(pose.qw, pose.qx, pose.qy, pose.qz)
    t[:3, 3] = [pose.x, pose.y, pose.z or 0.0]
    return t


def lever_arm(sensors: dict | None) -> np.ndarray:
    """T_pose_ego: where the cloud's sensor sits in the pose unit's frame. Identity unless the session says."""
    from services.lidar.calib.kitti import velo_pose_in_imu

    return velo_pose_in_imu((sensors or {}).get("imu_to_velo"))


def yaw_of(t: np.ndarray) -> float:
    return math.atan2(float(t[1, 0]), float(t[0, 0]))


def transform_points(t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return xyz @ t[:3, :3].T + t[:3, 3]


def world_cuboid_to_ego(center_world, yaw_world: float, t_world_ego: np.ndarray) -> tuple[list[float], float]:
    """A world-frame cuboid (centre, yaw about up) in one scan's ego frame."""
    inv = np.linalg.inv(t_world_ego)
    c = transform_points(inv, np.asarray(center_world, dtype=np.float64))[0]
    yaw = yaw_world - yaw_of(t_world_ego)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return [round(float(v), 4) for v in c], round(yaw, 5)


def ego_cuboid_to_world(center_ego, yaw_ego: float, t_world_ego: np.ndarray) -> tuple[list[float], float]:
    c = transform_points(t_world_ego, np.asarray(center_ego, dtype=np.float64))[0]
    yaw = yaw_ego + yaw_of(t_world_ego)
    return [round(float(v), 4) for v in c], round(math.atan2(math.sin(yaw), math.cos(yaw)), 5)


async def session_scene_poses(db: AsyncSession, session_id: uuid.UUID) -> tuple[list[ScenePose], dict]:
    """Every cloud of a session that a pose places, in time order, plus a summary of what could not be placed."""
    from db.models import EgoPose, PointCloud
    from db.models import Session as DbSession

    sess = await db.get(DbSession, session_id)
    if sess is None:
        return [], {"error": "session not found"}
    arm = lever_arm(sess.sensors)
    clouds = (await db.execute(select(PointCloud.cloud_id, PointCloud.ts_ns, PointCloud.cloud_uri)
                               .where(PointCloud.session_id == session_id)
                               .order_by(PointCloud.ts_ns))).all()
    poses = (await db.execute(select(EgoPose).where(EgoPose.session_id == session_id)
                              .order_by(EgoPose.ts_ns))).scalars().all()
    if not poses:
        return [], {"clouds": len(clouds), "placed": 0,
                    "reason": "this session has no ego poses, so its scans cannot share one frame"}
    pts = np.array([p.ts_ns for p in poses], dtype=np.int64)
    out: list[ScenePose] = []
    unplaced = 0
    for cid, ts, uri in clouds:
        i = int(np.searchsorted(pts, ts))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(poses) and abs(int(pts[j]) - int(ts)) <= POSE_MATCH_NS:
                if best is None or abs(int(pts[j]) - int(ts)) < abs(int(pts[best]) - int(ts)):
                    best = j
        if best is None:
            unplaced += 1
            continue
        out.append(ScenePose(cloud_id=cid, ts_ns=int(ts), cloud_uri=uri,
                             T_world_ego=pose_matrix(poses[best]) @ arm, pose_ts_ns=int(pts[best])))
    summary = {"clouds": len(clouds), "placed": len(out), "unplaced": unplaced,
               "lever_arm": bool((sess.sensors or {}).get("imu_to_velo")),
               "ego_frame": (sess.sensors or {}).get("ego_frame", "ego"),
               "lidar_height_m": (sess.sensors or {}).get("lidar_height_m")}
    return out, summary


def _voxel_down(xyz: np.ndarray, inten: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0 or voxel <= 0:
        return xyz, inten
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return xyz[first], inten[first]


def accumulate_scene(poses: list[ScenePose], *, every: int = 5, max_range_m: float = 40.0,
                     voxel_m: float = 0.15, max_points: int = 600_000, loader=None) -> dict:
    """Every `every`-th scan placed in the world frame, voxel-thinned, as one static cloud.

    Points beyond `max_range_m` of their own scan's sensor are dropped: a 64-beam scan at 80 m is a handful of
    rings, and stacking those rings from sixty scans draws streaks rather than surfaces.
    """
    from services.lidar.ingest.store import load_cloud

    loader = loader or load_cloud
    xyzs, intens, used = [], [], 0
    for sp in poses[::max(1, every)]:
        c = loader(sp.cloud_uri)
        xyz = np.asarray(c.xyz, dtype=np.float64)
        keep = np.hypot(xyz[:, 0], xyz[:, 1]) <= max_range_m
        w = transform_points(sp.T_world_ego, xyz[keep])
        it = np.asarray(c.intensity, dtype=np.float32)[keep] if c.intensity is not None else np.zeros(int(keep.sum()), np.float32)
        w, it = _voxel_down(w, it, voxel_m)
        xyzs.append(w)
        intens.append(it)
        used += 1
    if not xyzs:
        return {"xyz": np.zeros((0, 3), np.float32), "intensity": np.zeros(0, np.float32), "scans": 0}
    xyz = np.concatenate(xyzs)
    inten = np.concatenate(intens)
    xyz, inten = _voxel_down(xyz, inten, voxel_m)
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, inten = xyz[idx], inten[idx]
    return {"xyz": xyz.astype(np.float32), "intensity": inten.astype(np.float32), "scans": used}
