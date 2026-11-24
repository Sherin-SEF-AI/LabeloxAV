"""LiDAR ingestion: every source (real LiDAR, pseudo-LiDAR, public datasets) normalizes to one Cloud, which
round-trips through MCAP and the object store and links to camera frames by the PPS ts_ns."""

from services.lidar.ingest.bev_frame import ingest_lidar_sweep
from services.lidar.ingest.mcap_pc import (
    POINTS_TOPIC,
    iter_pointcloud_messages_mcap,
    pointcloud_topics_mcap,
    read_pointclouds_mcap,
    write_pointclouds_mcap,
)
from services.lidar.ingest.normalize import Cloud
from services.lidar.ingest.nuscenes_sweeps import (
    NuScenesLidarSweep,
    iter_nuscenes_lidar_clouds,
    list_nuscenes_lidar_sweeps,
    read_nuscenes_sweep,
)
from services.lidar.ingest.pointcloud2 import (
    PointCloud2,
    PointField,
    decode_pointcloud2,
    parse_pointcloud2_cdr,
    parse_pointcloud2_ros1,
)
from services.lidar.ingest.pseudo import (
    backproject,
    camera_to_ego,
    estimate_depth,
    lift_frame_group,
    place_in_world,
)
from services.lidar.ingest.readers import (
    read_kitti_bin,
    read_las,
    read_nuscenes_bin,
    read_pcd,
    read_point_cloud,
)
from services.lidar.ingest.rosbag1 import iter_pointcloud_messages_rosbag, read_pointclouds_rosbag
from services.lidar.ingest.store import load_cloud, store_cloud, store_derived

__all__ = [
    "Cloud",
    "read_point_cloud", "read_kitti_bin", "read_nuscenes_bin", "read_pcd", "read_las",
    "write_pointclouds_mcap", "read_pointclouds_mcap", "POINTS_TOPIC",
    "iter_pointcloud_messages_mcap", "pointcloud_topics_mcap",
    "PointCloud2", "PointField", "decode_pointcloud2", "parse_pointcloud2_cdr", "parse_pointcloud2_ros1",
    "read_pointclouds_rosbag", "iter_pointcloud_messages_rosbag",
    "NuScenesLidarSweep", "list_nuscenes_lidar_sweeps", "read_nuscenes_sweep", "iter_nuscenes_lidar_clouds",
    "store_cloud", "load_cloud", "store_derived",
    "lift_frame_group", "estimate_depth", "backproject", "camera_to_ego", "place_in_world",
    "ingest_lidar_sweep",
]
