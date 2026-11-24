"""The LiDAR sweeps of a standard nuScenes directory, as the ingest path wants them.

`list_nuscenes_lidar_sweeps` reads the tables (no devkit) and returns one record per LIDAR_TOP sample_data
row: where the `.pcd.bin` is, when it was captured, and the two rigid transforms that place it, sensor ->
ego (calibrated_sensor) and ego -> global (the ego_pose of that sweep). `iter_nuscenes_lidar_clouds` then
reads each file into the one internal Cloud, by default moved into the ego frame, which is the frame the
rest of the 3D pipeline works in (nuScenes' ego frame is x forward, y left, z up, like this codebase's; its
LiDAR frame is not, it has y forward).

Neither function opens a database or the object store. Persisting a cloud is services.lidar.ingest.store's
`store_cloud(cloud, session_id)`, which needs the session the camera frames were imported into.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.imports.nuscenes_tables import NuScenesTables, apply_pose, load_tables
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.readers import read_nuscenes_bin

LIDAR_CHANNEL = "LIDAR_TOP"


@dataclass
class NuScenesLidarSweep:
    path: Path                       # the .pcd.bin (float32 x, y, z, intensity, ring), not checked to exist
    ts_ns: int                       # capture time of this sweep, UTC nanoseconds
    sensor_to_ego: np.ndarray        # 4x4: p_ego = T @ p_sensor
    ego_to_global: np.ndarray        # 4x4: p_global = T @ p_ego, the ego pose at this sweep's capture time
    is_key_frame: bool               # True for the annotated sweep of a sample
    sample_token: str                # the sample this sweep belongs to (a non-keyframe sweep: the next one)
    sample_ts_ns: int | None         # that sample's timestamp: the ts_ns its camera frames are imported at
    token: str                       # the sample_data token
    scene_token: str | None = None
    channel: str = LIDAR_CHANNEL

    @property
    def sensor_to_global(self) -> np.ndarray:
        return self.ego_to_global @ self.sensor_to_ego


def list_nuscenes_lidar_sweeps(root: str | Path, channel: str = LIDAR_CHANNEL,
                               keyframes_only: bool = False,
                               tables: NuScenesTables | None = None) -> list[NuScenesLidarSweep]:
    """Every sweep of one LiDAR channel in a nuScenes directory, in time order.

    `keyframes_only` keeps the 2 Hz annotated sweeps and drops the 20 Hz ones between them. A row whose
    calibrated sensor or ego pose is missing from the tables is skipped: without them the cloud cannot be
    placed.
    """
    tables = tables or load_tables(Path(root))
    out: list[NuScenesLidarSweep] = []
    for sd in tables.sample_data:
        cs = tables.calibrated_sensor.get(sd.get("calibrated_sensor_token", ""))
        sensor = tables.sensor.get(cs.get("sensor_token", "")) if cs else None
        if sensor is None or sensor.get("modality") != "lidar" or sensor.get("channel") != channel:
            continue
        if sd.get("ego_pose_token") not in tables.ego_pose:
            continue
        key = bool(sd.get("is_key_frame"))
        if keyframes_only and not key:
            continue
        sample = tables.sample.get(sd.get("sample_token", ""), {})
        out.append(NuScenesLidarSweep(
            path=tables.file_path(sd), ts_ns=int(sd["timestamp"]) * 1000,
            sensor_to_ego=tables.sensor_to_ego(sd), ego_to_global=tables.ego_to_global(sd),
            is_key_frame=key, sample_token=sd.get("sample_token", ""),
            sample_ts_ns=int(sample["timestamp"]) * 1000 if "timestamp" in sample else None,
            token=sd.get("token", ""), scene_token=sample.get("scene_token"), channel=channel,
        ))
    out.sort(key=lambda s: (s.ts_ns, s.token))
    return out


def read_nuscenes_sweep(sweep: NuScenesLidarSweep, frame: str = "ego") -> Cloud:
    """Read one sweep's file into a Cloud in the `sensor`, `ego` or `global` frame.

    A keyframe sweep is stamped with its sample's timestamp, the ts_ns the sample's camera frames are
    imported at (services.imports.adapter_nuscenes), so the cloud and those frames are one query; any other
    sweep keeps its own capture time. `global` coordinates are stored as float32 like every Cloud, which at
    nuScenes map scale (a couple of kilometres) is sub-millimetre.
    """
    if frame not in ("sensor", "ego", "global"):
        raise ValueError(f"frame must be 'sensor', 'ego' or 'global', got {frame!r}")
    ts_ns = sweep.sample_ts_ns if (sweep.is_key_frame and sweep.sample_ts_ns is not None) else sweep.ts_ns
    cloud = read_nuscenes_bin(Path(sweep.path).read_bytes(), ts_ns)
    if frame == "sensor":
        return cloud
    pose = sweep.sensor_to_ego if frame == "ego" else sweep.sensor_to_global
    return Cloud(xyz=apply_pose(pose, cloud.xyz), intensity=cloud.intensity, ts_ns=cloud.ts_ns,
                 ring=cloud.ring, source=cloud.source, frame="ego" if frame == "ego" else "world")


def iter_nuscenes_lidar_clouds(root: str | Path, channel: str = LIDAR_CHANNEL, keyframes_only: bool = True,
                               frame: str = "ego") -> Iterator[tuple[NuScenesLidarSweep, Cloud]]:
    """Yield (sweep, Cloud) for each sweep whose file is present, in time order. Missing files are skipped,
    as a nuScenes download is commonly a subset of the blobs its tables list."""
    for sweep in list_nuscenes_lidar_sweeps(root, channel, keyframes_only):
        if not Path(sweep.path).is_file():
            continue
        yield sweep, read_nuscenes_sweep(sweep, frame)
