"""3D detection: 2D-to-3D lifting (primary for the camera fleet) and native detection (real LiDAR, via the
burst seam), both passing through the same governed gate and ontology."""

from services.lidar.detect3d.fuse3d import gate_cuboid
from services.lidar.detect3d.lift import fit_cuboid, frustum_indices, lift_box
from services.lidar.detect3d.native import (
    NativeDetectionUnavailable,
    detect_native,
    native_available,
    native_class_to_ontology,
)
from services.lidar.detect3d.run import detect_native_cloud, lift_frame

__all__ = [
    "frustum_indices", "fit_cuboid", "lift_box", "gate_cuboid",
    "detect_native", "native_available", "native_class_to_ontology", "NativeDetectionUnavailable",
    "lift_frame", "detect_native_cloud",
]
