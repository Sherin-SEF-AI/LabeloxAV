"""M-CAL.1: the calibration seam. The nominal Calibration must reproduce the legacy projection EXACTLY (so
swapping every 3D consumer onto the resolver changes nothing until real calibration is stored), full 6-DOF
extrinsics must generalize the legacy yaw+height, and the resolver must fall back to nominal when a session
has no stored calibration."""

from __future__ import annotations

import uuid

import numpy as np

from services.calibration.resolve import (
    SOURCE_QUALITY,
    Calibration,
    calibration_from_row,
    nominal_calibration,
    resolve_calibration,
)

W, H = 1920, 1080


def test_nominal_reproduces_legacy_intrinsics_and_extrinsics():
    from services.lidar.project import _ego_to_camera_matrix, _intrinsics
    for cam in ["cam_f", "cam_l", "cam_r", "cam_b"]:
        c = nominal_calibration(cam, W, H)
        k, fx, fy, cx, cy = _intrinsics(cam, W, H)
        assert abs(c.fx - fx) < 1e-6 and abs(c.fy - fy) < 1e-6
        assert abs(c.cx - cx) < 1e-6 and abs(c.cy - cy) < 1e-6
        assert c.model == k.model
        m, height = _ego_to_camera_matrix(cam)
        assert np.allclose(c.R(), m, atol=1e-5), f"{cam} extrinsic rotation must match legacy"
        assert np.allclose(c.t(), [0.0, 0.0, height], atol=1e-6)


def test_nominal_source_and_quality():
    c = nominal_calibration("cam_f", W, H)
    assert c.source == "nominal" and c.quality == SOURCE_QUALITY["nominal"]


def test_K_matrix_layout():
    c = nominal_calibration("cam_f", W, H)
    k = c.K()
    assert k[0, 0] == c.fx and k[1, 1] == c.fy and k[0, 2] == c.cx and k[1, 2] == c.cy and k[2, 2] == 1.0


def test_real_6dof_extrinsics_differ_from_nominal():
    real = Calibration("cam_f", "pinhole", 2870, 2870, 960, 540, [],
                       rpy_deg=(0.0, 10.0, 0.0), xyz_m=(1.5, 0.0, 1.6), source="measured", quality=1.0)
    nom = nominal_calibration("cam_f", 1920, 1080)
    assert not np.allclose(real.R(), nom.R())   # a real downward pitch rotates the optical axis
    assert not np.allclose(real.t(), nom.t())   # a real mount offset translates it


def test_calibration_from_row_scales_to_image():
    class Row:
        cam_id = "cam_f"
        model = "pinhole"
        fx = fy = 2870.0
        cx = 960.0
        cy = 540.0
        dist: list = []
        ref_width = 1920
        rpy_deg = [0.0, 0.0, 0.0]
        xyz_m = [0.0, 0.0, 1.5]
        source = "dataset"
        quality = 0.9
    c = calibration_from_row(Row(), 960, 540)   # half the reference width
    assert abs(c.fx - 1435.0) < 1e-3 and abs(c.cx - 480.0) < 1e-3   # intrinsics and principal point halved
    assert c.source == "dataset" and c.quality == 0.9


async def test_resolver_falls_back_to_nominal_when_unstored():
    c = await resolve_calibration(uuid.uuid4(), "cam_f", W, H)   # a random session has no stored calibration
    assert c.source == "nominal"
    assert abs(c.fx - nominal_calibration("cam_f", W, H).fx) < 1e-6
