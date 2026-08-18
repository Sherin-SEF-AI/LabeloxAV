"""The calibration a session uses is the calibration its geometry is built from, and validated against.

`project_to_camera` grew a `calib` parameter (M-CAL.1) and two call sites were left behind, both of them
load-bearing:

  - `detect3d/lift.py` is the primary cuboid source for the camera fleet. `frustum_indices` had no calib
    parameter at all, so every lifted 3D box on a calibrated session was still being cut out of the
    config-declared rig, and every metric downstream inherited that error.
  - `calib/lidar_camera.py` is the validator. Measuring the reprojection residual through the nominal rig
    asks whether the points sit where the CONFIG says the camera is - a question a drifted stored
    calibration passes trivially, because the thing under test is never consulted.

These pin both. They are pure geometry: a synthetic point cloud and two calibrations, no database.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from services.calibration.resolve import nominal_calibration
from services.lidar.calib.lidar_camera import reprojection_error
from services.lidar.detect3d.lift import frustum_indices
from services.lidar.project import project_to_camera

CAM, W, H = "cam_f", 1280, 960


def _drifted(cam_id: str, w: int, h: int, yaw_deg: float = 4.0):
    """The nominal calibration with the camera mount yawed a few degrees, standing in for a real drift.

    Drifts `rpy_deg` rather than any cached matrix: `Calibration.R()` derives the rotation from the mount
    angles every call, so replacing a matrix attribute would leave the projection unchanged and quietly
    turn every assertion below into a tautology.
    """
    calib = nominal_calibration(cam_id, w, h)
    roll, pitch, yaw = calib.rpy_deg
    return replace(calib, rpy_deg=(roll, pitch, yaw + yaw_deg))


class TestTheFrustumUsesTheCalibrationItWasGiven:
    def test_frustum_indices_accepts_a_calibration(self):
        # The parameter did not exist. Its absence is the bug, so its presence is worth asserting directly.
        import inspect

        assert "calib" in inspect.signature(frustum_indices).parameters

    def test_a_different_calibration_selects_different_points(self):
        """If calib were ignored, these two frustums would be identical and the whole thread would be inert."""
        rng = np.random.default_rng(20260818)
        cloud = np.column_stack([
            rng.uniform(5.0, 30.0, 4000),      # x forward
            rng.uniform(-8.0, 8.0, 4000),      # y left
            rng.uniform(-1.0, 2.0, 4000),      # z up
        ]).astype(np.float32)
        box = [500.0, 400.0, 800.0, 700.0]

        nominal = frustum_indices(cloud, box, CAM, W, H)
        drifted = frustum_indices(cloud, box, CAM, W, H, calib=_drifted(CAM, W, H))

        assert len(nominal) > 50, "fixture is not exercising the frustum at all"
        assert set(nominal.tolist()) != set(drifted.tolist()), (
            "the same points fell in the frustum under two different calibrations, so calib is being ignored")

    def test_passing_none_reproduces_the_previous_behaviour(self):
        rng = np.random.default_rng(7)
        cloud = np.column_stack([rng.uniform(5.0, 30.0, 500), rng.uniform(-8.0, 8.0, 500),
                                 rng.uniform(-1.0, 2.0, 500)]).astype(np.float32)
        box = [0.0, 0.0, float(W), float(H)]
        explicit_nominal = frustum_indices(cloud, box, CAM, W, H, calib=nominal_calibration(CAM, W, H))
        assert set(frustum_indices(cloud, box, CAM, W, H).tolist()) == set(explicit_nominal.tolist())


class TestTheValidatorValidatesTheCalibrationInUse:
    def test_a_drifted_calibration_does_not_pass_its_own_check(self):
        """The inversion, stated as a test.

        Observations are generated through the drifted calibration - that is what the camera saw. Scoring
        them against the nominal rig (calib=None, the old behaviour) reports a large residual for a
        calibration that is in fact describing those pixels perfectly; scoring them against the calibration
        in use reports ~0. The old code did the former and called it validation.
        """
        rng = np.random.default_rng(99)
        pts = np.column_stack([rng.uniform(8.0, 25.0, 300), rng.uniform(-5.0, 5.0, 300),
                               rng.uniform(-0.5, 1.5, 300)]).astype(np.float32)
        drifted = _drifted(CAM, W, H)

        observed = project_to_camera(pts, CAM, W, H, drifted)["uv"]

        against_the_real_one = reprojection_error(pts, observed, CAM, W, H, drifted)
        against_the_config = reprojection_error(pts, observed, CAM, W, H)

        assert against_the_real_one["n"] > 0
        assert against_the_real_one["rms"] < 1e-3, (
            "points projected through a calibration must reproject through it with ~no residual")
        assert against_the_config["rms"] > 1.0, (
            "fixture is not drifted enough for this comparison to mean anything")

    def test_the_residual_is_zero_when_the_calibration_describes_the_pixels(self):
        rng = np.random.default_rng(3)
        pts = np.column_stack([rng.uniform(8.0, 25.0, 100), rng.uniform(-5.0, 5.0, 100),
                               rng.uniform(-0.5, 1.5, 100)]).astype(np.float32)
        nominal = nominal_calibration(CAM, W, H)
        observed = project_to_camera(pts, CAM, W, H, nominal)["uv"]
        assert reprojection_error(pts, observed, CAM, W, H, nominal)["rms"] < 1e-3
