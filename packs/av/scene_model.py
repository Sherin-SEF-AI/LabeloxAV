"""AV scene model: a moving camera on a vehicle.

Wraps the resolved per-camera Calibration (services/calibration/resolve.py) - the ego-frame optical convention
the audit identified as the moving-vs-static fork. The reference frame is the ego (vehicle) frame; the full
world<-camera pose composes the vehicle's world pose at the frame time with the fixed ego<-camera mount, which
is exactly what the legacy projection did. There is a road ground plane (z=0 in the ego frame).
"""

from __future__ import annotations

import numpy as np

from core.geometry import Plane, se3, se3_compose
from services.calibration.resolve import Calibration


class MovingCameraSceneModel:
    def __init__(self, calib: Calibration) -> None:
        self._calib = calib

    def is_static(self) -> bool:
        return False

    def rotation(self) -> np.ndarray:
        """The legacy rotation convention: cam = (ego - t) @ R (row vectors)."""
        return self._calib.R()

    def extrinsic(self) -> np.ndarray:
        """SE(3) ego<-camera. From cam = (ego - t) @ R it follows p_ego = R @ p_cam + t, so the mount pose is
        se3(R, t) with R = Calibration.R() and t = the mount xyz."""
        return se3(self._calib.R().astype(np.float64), self._calib.t().astype(np.float64))

    def camera_pose(self, world_from_reference: np.ndarray | None = None) -> np.ndarray:
        """SE(3) world<-camera = world<-ego(frame) o ego<-camera. Without a per-frame ego pose the ego frame is
        the reference (identity), reproducing the single-frame projection."""
        ext = self.extrinsic()
        if world_from_reference is None:
            return ext
        return se3_compose(np.asarray(world_from_reference, dtype=np.float64), ext)

    def ground_plane(self) -> Plane | None:
        """The road plane in the ego frame: z=0 with the normal up. The camera sits at xyz_m[2] above it; the
        IPM / ground-contact depth priors read this plane plus the mount height."""
        return Plane(normal=(0.0, 0.0, 1.0), offset=0.0, frame="ego")


class MovingCameraSceneModelFactory:
    def build(self, calib: Calibration) -> MovingCameraSceneModel:
        return MovingCameraSceneModel(calib)
