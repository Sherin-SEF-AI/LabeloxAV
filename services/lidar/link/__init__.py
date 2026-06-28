"""Unify 2D and 3D into one identity, auto-compute object properties, and batch-correct similar objects."""

from services.lidar.link.correct import batch_correct, find_similar
from services.lidar.link.object_identity import (
    assign_object_id,
    link_cloud,
    linked_from_2d,
    linked_views,
    projected_bbox,
)
from services.lidar.link.properties import compute_object_properties

__all__ = [
    "link_cloud", "linked_views", "linked_from_2d", "assign_object_id", "projected_bbox",
    "compute_object_properties", "find_similar", "batch_correct",
]
