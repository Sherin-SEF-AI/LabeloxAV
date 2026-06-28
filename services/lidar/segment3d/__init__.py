"""Point cloud segmentation: PTv3 semantic and PTv3+PointGroup instance (burst seam), plus a runnable
projected segmentation for the camera fleet that labels points from the 3D cuboids and the ground, flagging
low-confidence regions for review."""

from services.lidar.segment3d.instance import segment_instance_ptv3
from services.lidar.segment3d.run import load_segmentation, segment_cloud
from services.lidar.segment3d.semantic import (
    SegmentationUnavailable,
    points_in_cuboid,
    road_class_id,
    segment_projected,
    segment_ptv3,
)

__all__ = [
    "segment_projected", "segment_ptv3", "points_in_cuboid", "road_class_id", "SegmentationUnavailable",
    "segment_instance_ptv3", "segment_cloud", "load_segmentation",
]
