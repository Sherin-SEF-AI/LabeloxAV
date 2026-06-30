"""3D quality checking and 3D scene intelligence: the mandatory 3D quality checker (floating, below ground,
impossible dims, duplicate, misaligned, missing neighbour) that feeds the same review loop as 2D, plus 3D
scene classification and 3D rare-event mining that extend the Phase 1 intelligence."""

from services.lidar.quality3d.checker import (
    check_cloud,
    check_cuboid,
    confirm_flag,
    find_missing_neighbors,
)
from services.lidar.quality3d.rare3d import mine_3d_cues, mine_session_3d
from services.lidar.quality3d.scene3d import classify_3d_structure, classify_session_3d

__all__ = [
    "check_cuboid", "find_missing_neighbors", "check_cloud", "confirm_flag",
    "classify_3d_structure", "classify_session_3d", "mine_3d_cues", "mine_session_3d",
]
