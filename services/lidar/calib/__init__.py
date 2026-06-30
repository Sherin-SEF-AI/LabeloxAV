"""LiDAR calibration validation: reprojection and consistency residuals against the camera, drift detection,
and the 3D-annotation exclusion gate for sessions that fail calibration or cloud quality."""

from services.lidar.calib.lidar_camera import coverage_consistency, reprojection_error
from services.lidar.calib.validate3d import (
    lidar_session_ok,
    record_validation,
    validate_lidar_camera,
    validate_session,
)

__all__ = [
    "reprojection_error", "coverage_consistency",
    "validate_lidar_camera", "validate_session", "lidar_session_ok", "record_validation",
]
