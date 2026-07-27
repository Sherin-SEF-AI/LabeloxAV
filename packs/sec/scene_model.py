"""Sec scene model: a fixed (static) CCTV camera.

The opposite pole of the fork the audit found at services/calibration/resolve.py. The camera does not move, so:

* the reference frame is the fixed world, not a moving ego frame;
* the world<-camera pose is constant - camera_pose() ignores any ego pose passed to it;
* there is no single road ground plane - ground_plane() is None;
* but there is a stable per-camera background, which a moving camera never has. The model can fit that prior
  (core/scene/background.py) and turn "what moved" into a model-free foreground signal.

state is keyed by the camera identity (Frame.cam_id), never a vehicle.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.geometry import Plane, se3
from core.scene.background import foreground_mask as _foreground_mask
from core.scene.background import temporal_median
from services.calibration.resolve import Calibration


class StaticCameraSceneModel:
    def __init__(self, calib: Calibration, background: np.ndarray | None = None) -> None:
        self._calib = calib
        self._background = background

    def is_static(self) -> bool:
        return True

    def rotation(self) -> np.ndarray:
        return self._calib.R()

    def extrinsic(self) -> np.ndarray:
        """SE(3) world<-camera. The camera is mounted directly in the fixed world, so the mount pose se3(R, t)
        IS the world pose - there is no ego frame in between."""
        return se3(self._calib.R().astype(np.float64), self._calib.t().astype(np.float64))

    def camera_pose(self, world_from_reference: np.ndarray | None = None) -> np.ndarray:
        """SE(3) world<-camera. Static: constant across frames; any ego pose passed in is ignored (there is no
        moving vehicle to compose)."""
        return self.extrinsic()

    def ground_plane(self) -> Plane | None:
        """A static scene has no single road plane (a corridor, a forecourt, a platform are not one ground
        plane the way a road ahead is), so there is none to expose."""
        return None

    # ---- static-only capability: the background prior --------------------------------------------------

    def fit_background(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Fit and store the per-camera background prior from a sample of frames (temporal median)."""
        self._background = temporal_median(frames)
        return self._background

    def background_prior(self) -> np.ndarray | None:
        return self._background

    def foreground_mask(self, frame: np.ndarray, threshold: float = 25.0) -> np.ndarray:
        """Pixels of `frame` that differ from the fitted background prior. Requires fit_background first."""
        if self._background is None:
            raise RuntimeError("no background prior; call fit_background() first")
        return _foreground_mask(frame, self._background, threshold)


class StaticCameraSceneModelFactory:
    def build(self, calib: Calibration) -> StaticCameraSceneModel:
        return StaticCameraSceneModel(calib)
