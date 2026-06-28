"""LiDAR ingestion: every source (real LiDAR, pseudo-LiDAR, public datasets) normalizes to one Cloud, which
round-trips through MCAP and the object store and links to camera frames by the PPS ts_ns."""

from services.lidar.ingest.bev_frame import ingest_lidar_sweep
from services.lidar.ingest.mcap_pc import (
    POINTS_TOPIC,
    read_pointclouds_mcap,
    write_pointclouds_mcap,
)
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.readers import (
    read_kitti_bin,
    read_las,
    read_nuscenes_bin,
    read_pcd,
    read_point_cloud,
)
from services.lidar.ingest.store import load_cloud, store_cloud, store_derived

__all__ = [
    "Cloud",
    "read_point_cloud", "read_kitti_bin", "read_nuscenes_bin", "read_pcd", "read_las",
    "write_pointclouds_mcap", "read_pointclouds_mcap", "POINTS_TOPIC",
    "store_cloud", "load_cloud", "store_derived",
    "ingest_lidar_sweep",
]
