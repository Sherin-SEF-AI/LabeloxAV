"""Point cloud readers for every supported source, each producing the one internal Cloud (normalize.py).

  real LiDAR / buyer clouds:  PCD (pypcd4), LAS and LAZ (laspy + lazrs)
  public datasets:            KITTI Velodyne .bin (N x 4), nuScenes .bin (N x 5 with ring)

read_point_cloud dispatches by extension. A .bin is KITTI (xyzi) or nuScenes (xyzir); the layout is inferred
from the float count and can be forced with `features`. Every reader returns metres in the sensor frame with
intensity carried through (normalized to 0..1 where the source range is known).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from services.lidar.ingest.normalize import Cloud


def read_kitti_bin(data: bytes, ts_ns: int = 0) -> Cloud:
    """KITTI Velodyne: a flat float32 buffer of (x, y, z, intensity). Intensity is already 0..1."""
    arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
    return Cloud(xyz=arr[:, :3], intensity=arr[:, 3], ts_ns=ts_ns, source="dataset", frame="kitti_velo")


def read_nuscenes_bin(data: bytes, ts_ns: int = 0) -> Cloud:
    """nuScenes LiDAR: float32 (x, y, z, intensity, ring). Intensity is 0..255, normalized to 0..1."""
    arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 5)
    return Cloud(xyz=arr[:, :3], intensity=arr[:, 3] / 255.0, ring=arr[:, 4].astype(np.int16),
                 ts_ns=ts_ns, source="dataset", frame="nuscenes_lidar")


def read_pcd(data: bytes, ts_ns: int = 0, source: str = "lidar") -> Cloud:
    """PCD (ascii, binary, or binary_compressed) via pypcd4. xyz is required; intensity and ring optional."""
    from pypcd4 import PointCloud as PCD

    pc = PCD.from_fileobj(io.BytesIO(data))
    fields = set(pc.fields)
    if not {"x", "y", "z"} <= fields:
        raise ValueError(f"PCD is missing xyz fields, has {sorted(fields)}")
    xyz = pc.numpy(("x", "y", "z")).astype(np.float32)
    intensity = pc.numpy(("intensity",)).reshape(-1).astype(np.float32) if "intensity" in fields else None
    ring_field = next((f for f in ("ring", "r") if f in fields), None)
    ring = pc.numpy((ring_field,)).reshape(-1).astype(np.int16) if ring_field else None
    return Cloud(xyz=xyz, intensity=intensity, ring=ring, ts_ns=ts_ns, source=source, frame="sensor")


def read_las(data: bytes, ts_ns: int = 0, source: str = "lidar") -> Cloud:
    """LAS and LAZ via laspy (LAZ through the lazrs backend). Intensity is 0..65535, normalized to 0..1."""
    import laspy

    las = laspy.read(io.BytesIO(data))
    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1).astype(np.float32)
    inten = np.asarray(las.intensity).astype(np.float32) / 65535.0 if "intensity" in las.point_format.dimension_names else None
    return Cloud(xyz=xyz, intensity=inten, ts_ns=ts_ns, source=source, frame="sensor")


def read_point_cloud(path: str | Path, ts_ns: int = 0, features: int | None = None) -> Cloud:
    """Dispatch by file extension. `features` forces a .bin layout (4 = KITTI xyzi, 5 = nuScenes xyzir)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    ext = path.suffix.lower()
    if ext in (".pcd",):
        return read_pcd(data, ts_ns)
    if ext in (".las", ".laz"):
        return read_las(data, ts_ns)
    if ext == ".bin":
        n_floats = len(data) // 4
        feats = features or _infer_bin_features(n_floats, path.name)
        return read_nuscenes_bin(data, ts_ns) if feats == 5 else read_kitti_bin(data, ts_ns)
    raise ValueError(f"unsupported point cloud format: {ext}")


def _infer_bin_features(n_floats: int, name: str) -> int:
    """A KITTI .bin is divisible by 4, nuScenes by 5. When both fit, the filename hints, else default KITTI."""
    div4, div5 = (n_floats % 4 == 0), (n_floats % 5 == 0)
    if div5 and not div4:
        return 5
    if div4 and not div5:
        return 4
    return 5 if "nuscenes" in name.lower() else 4
