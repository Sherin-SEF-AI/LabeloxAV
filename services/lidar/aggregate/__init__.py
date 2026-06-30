"""Multi-scan alignment, loop closure, and aggregation into a dense map."""

from services.lidar.aggregate.accumulate import accumulate_scans, transform_cloud
from services.lidar.aggregate.loopclose import detect_loops, optimize_pose_graph
from services.lidar.aggregate.register import accumulate_poses, gnss_imu_prior, register_pair
from services.lidar.aggregate.run import aggregate_sessions

__all__ = [
    "register_pair", "gnss_imu_prior", "accumulate_poses",
    "detect_loops", "optimize_pose_graph",
    "accumulate_scans", "transform_cloud", "aggregate_sessions",
]
