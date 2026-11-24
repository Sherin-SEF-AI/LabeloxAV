"""Reading a standard nuScenes table directory, and the rigid-transform math it needs, without the devkit.

A nuScenes release is a directory of JSON tables (`v1.0-mini/`, `v1.0-trainval/`) next to the `samples/`
and `sweeps/` folders its `sample_data.filename` values point into. Every pose in it is a translation plus a
unit quaternion in `[w, x, y, z]` order, and each one is the pose OF the child frame IN the parent frame:

    calibrated_sensor   sensor -> ego      p_ego    = R_cs  @ p_sensor + t_cs
    ego_pose            ego    -> global   p_global = R_ego @ p_ego    + t_ego
    sample_annotation   box    -> global   box x is the length (front), y the width (left), z the height;
                                           `size` is [width, length, height]

The camera frame is the optical one (x right, y down, z forward) and `camera_intrinsic` is the 3x3 pinhole
matrix K, so a point in the camera frame lands on pixel `(K @ p)[:2] / p[2]`.

Pure numpy and json: no database, no object store, nothing imported from the rest of the import driver, so
both the 2D import adapter (services.imports.adapter_nuscenes) and the LiDAR ingest path
(services.lidar.ingest.nuscenes_sweeps) can use it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Tables the standard import cannot work without. attribute, visibility, scene and log are optional: a
# directory without them still imports, with the fields they would have filled left empty.
REQUIRED_TABLES = ("sample", "sample_data", "sample_annotation", "instance", "category",
                   "calibrated_sensor", "ego_pose", "sensor")
OPTIONAL_TABLES = ("attribute", "visibility", "scene", "log")


# --- quaternions and rigid transforms ---------------------------------------------------------------------

def _unit(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError(f"not a rotation quaternion: {q.tolist()}")
    return q / n


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a `[w, x, y, z]` quaternion (Hamilton convention; normalized first)."""
    w, x, y, z = _unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quats_to_matrices(q) -> np.ndarray:
    """`quat_to_matrix` for an (N, 4) array at once, returning (N, 3, 3)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1)
    if not np.all(np.isfinite(n)) or np.any(n < 1e-12):
        raise ValueError("a rotation quaternion has zero or non-finite length")
    w, x, y, z = (q / n[:, None]).T
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - w * z)
    out[:, 0, 2] = 2 * (x * z + w * y)
    out[:, 1, 0] = 2 * (x * y + w * z)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - w * x)
    out[:, 2, 0] = 2 * (x * z - w * y)
    out[:, 2, 1] = 2 * (y * z + w * x)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product a * b of two `[w, x, y, z]` quaternions: the rotation b followed by a."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_conjugate(q) -> np.ndarray:
    """The inverse rotation of a unit quaternion."""
    w, x, y, z = _unit(q)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_canonical(q) -> np.ndarray:
    """Unit length with w >= 0, so the two quaternions that name one rotation compare equal."""
    q = _unit(q)
    return -q if q[0] < 0 else q


def heading_of(rotation: np.ndarray) -> float:
    """Yaw about the frame's z axis of a body whose x axis is its front: the angle of the rotated x axis
    projected onto the xy plane. For a box that is level in the frame this is exactly its yaw; for a tilted
    one it is the heading a bird's-eye view shows."""
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def pose_matrix(rotation, translation) -> np.ndarray:
    """4x4 homogeneous transform of a nuScenes (rotation quaternion, translation) pose: child -> parent."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_matrix(rotation)
    out[:3, 3] = t
    return out


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (parent -> child), without a general matrix inverse."""
    r, t = pose[:3, :3], pose[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def apply_pose(pose: np.ndarray, points) -> np.ndarray:
    """Transform (N, 3) points by a 4x4 rigid transform."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ pose[:3, :3].T + pose[:3, 3]


# --- tables -----------------------------------------------------------------------------------------------

@dataclass
class NuScenesTables:
    """The tables of one nuScenes version directory, indexed by token."""

    table_dir: Path                     # the directory holding the *.json tables
    dataroot: Path                      # its parent: what sample_data.filename is relative to
    sample: dict[str, dict]
    sample_data: list[dict]
    sample_annotation: list[dict]
    instance: dict[str, dict]
    category: dict[str, dict]
    calibrated_sensor: dict[str, dict]
    ego_pose: dict[str, dict]
    sensor: dict[str, dict]
    attribute: dict[str, dict] = field(default_factory=dict)
    visibility: dict[str, dict] = field(default_factory=dict)
    scene: dict[str, dict] = field(default_factory=dict)
    log: dict[str, dict] = field(default_factory=dict)

    def sensor_of(self, sample_data_row: dict) -> dict:
        """The sensor row ({channel, modality}) a sample_data row was captured by."""
        cs = self.calibrated_sensor[sample_data_row["calibrated_sensor_token"]]
        return self.sensor[cs["sensor_token"]]

    def sensor_to_ego(self, sample_data_row: dict) -> np.ndarray:
        cs = self.calibrated_sensor[sample_data_row["calibrated_sensor_token"]]
        return pose_matrix(cs["rotation"], cs["translation"])

    def ego_to_global(self, sample_data_row: dict) -> np.ndarray:
        ep = self.ego_pose[sample_data_row["ego_pose_token"]]
        return pose_matrix(ep["rotation"], ep["translation"])

    def file_path(self, sample_data_row: dict) -> Path:
        """Where the row's file should be on disk (not checked for existence)."""
        return self.dataroot / sample_data_row["filename"]


def find_table_dir(root: Path) -> Path | None:
    """The directory under root holding sample_data.json. When a download carries several versions
    (v1.0-mini beside v1.0-trainval) the first in sorted order is used; pass the version directory itself
    as root to choose another."""
    root = Path(root)
    if (root / "sample_data.json").is_file():
        return root
    hits = sorted(root.rglob("sample_data.json"))
    return hits[0].parent if hits else None


def _rows(table_dir: Path, name: str, required: bool) -> list[dict]:
    path = table_dir / f"{name}.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"nuScenes table {name}.json is missing from {table_dir}")
        return []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"nuScenes table {name}.json is not a list of rows")
    return rows


def _by_token(rows: list[dict]) -> dict[str, dict]:
    return {r["token"]: r for r in rows}


def load_tables(root: Path, table_dir: Path | None = None) -> NuScenesTables:
    """Load one nuScenes version directory found under root (or the given table_dir)."""
    table_dir = Path(table_dir) if table_dir is not None else find_table_dir(Path(root))
    if table_dir is None:
        raise FileNotFoundError("no nuScenes sample_data.json found under the dataset")
    req = {name: _rows(table_dir, name, required=True) for name in REQUIRED_TABLES}
    opt = {name: _rows(table_dir, name, required=False) for name in OPTIONAL_TABLES}
    return NuScenesTables(
        table_dir=table_dir, dataroot=table_dir.parent,
        sample=_by_token(req["sample"]), sample_data=req["sample_data"],
        sample_annotation=req["sample_annotation"], instance=_by_token(req["instance"]),
        category=_by_token(req["category"]), calibrated_sensor=_by_token(req["calibrated_sensor"]),
        ego_pose=_by_token(req["ego_pose"]), sensor=_by_token(req["sensor"]),
        attribute=_by_token(opt["attribute"]), visibility=_by_token(opt["visibility"]),
        scene=_by_token(opt["scene"]), log=_by_token(opt["log"]),
    )
