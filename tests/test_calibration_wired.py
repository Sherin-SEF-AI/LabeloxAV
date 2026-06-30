"""M-CAL.2: the resolver is wired into projection. calib=None must reproduce the legacy nominal projection
exactly (so nothing changes until real calibration is stored), and passing a real Calibration must actually
change where points land (so stored calibration flows through to every metric-3D consumer)."""

from __future__ import annotations

import numpy as np

from services.calibration.resolve import Calibration, nominal_calibration
from services.lidar.project import project_to_camera

W, H = 1280, 960
PTS = np.array([[10.0, 0.0, 0.0], [8.0, 2.0, 0.5], [15.0, -3.0, 1.0]], dtype=np.float32)


def test_calib_none_equals_explicit_nominal():
    legacy = project_to_camera(PTS, "cam_f", W, H)                                    # nominal (legacy) branch
    resolved = project_to_camera(PTS, "cam_f", W, H, nominal_calibration("cam_f", W, H))  # resolver branch
    assert np.allclose(legacy["uv"], resolved["uv"], atol=1e-3)
    assert np.array_equal(legacy["in_image"], resolved["in_image"])
    assert np.allclose(legacy["depth"], resolved["depth"], atol=1e-4)


def test_real_calibration_shifts_the_projection():
    nom = nominal_calibration("cam_f", W, H)                     # cam_f is a pinhole lens
    real = Calibration("cam_f", nom.model, nom.fx, nom.fy, nom.cx + 120.0, nom.cy - 80.0,
                       nom.dist, nom.rpy_deg, nom.xyz_m, source="measured", quality=1.0)
    base = project_to_camera(PTS, "cam_f", W, H)
    moved = project_to_camera(PTS, "cam_f", W, H, real)
    front = base["in_front"]
    # a pinhole principal-point shift moves every in-front pixel by exactly that delta
    assert np.allclose(moved["uv"][front] - base["uv"][front], [120.0, -80.0], atol=1e-2)
