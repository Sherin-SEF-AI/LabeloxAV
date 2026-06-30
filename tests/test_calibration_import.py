"""M-CAL.3c: explicit calib-file import. The intrinsics a dataset ships (KITTI P2, nuScenes
camera_intrinsic) are parsed exactly and stored as source=dataset, then read back by the resolver, a real
focal and principal point in place of the nominal lens."""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from db.models import CameraCalibration
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calibration.import_calib import (
    import_calibration,
    intrinsics_from_K,
    parse_kitti_calib,
    parse_nuscenes_calib,
)
from services.calibration.resolve import resolve_calibration
from services.calibration.store import upsert_calibration

_CAM = "cam_import_mcal3"

_KITTI = """P0: 7.2e+02 0 6.0e+02 0 0 7.2e+02 1.7e+02 0 0 0 1 0
P2: 721.5 0.0 609.6 44.9 0.0 721.5 172.9 0.2 0.0 0.0 1.0 0.003
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
"""


def test_intrinsics_from_K():
    f = intrinsics_from_K([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
    assert f["fx"] == 1000.0 and f["fy"] == 1000.0 and f["cx"] == 640.0 and f["cy"] == 360.0


def test_parse_kitti_p2():
    f = parse_kitti_calib(_KITTI)
    assert abs(f["fx"] - 721.5) < 1e-3 and abs(f["fy"] - 721.5) < 1e-3
    assert abs(f["cx"] - 609.6) < 1e-3 and abs(f["cy"] - 172.9) < 1e-3


def test_parse_kitti_missing_line_raises():
    with pytest.raises(ValueError):
        parse_kitti_calib("Tr_velo_to_cam: 0 0 0\n", cam="P2")


def test_parse_nuscenes_with_translation():
    k = [[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]]
    f = parse_nuscenes_calib(k, translation=[1.7, 0.0, 1.55])
    assert abs(f["fx"] - 1266.4) < 1e-3 and abs(f["cx"] - 816.3) < 1e-3
    assert f["xyz_m"] == [1.7, 0.0, 1.55]


async def test_import_then_resolve_reads_dataset_intrinsics():
    async with get_sessionmaker()() as db:
        sid = (await db.execute(select(DbSession.session_id).limit(1))).scalar()
    assert sid is not None
    try:
        intr = parse_kitti_calib(_KITTI)
        res = await import_calibration(sid, _CAM, intr, ref_width=1242)
        assert res["stored"]
        c = await resolve_calibration(sid, _CAM, 1242, 375)
        assert c.source == "dataset"
        assert abs(c.fx - 721.5) < 1e-2          # the real KITTI focal at native width, not nominal
        # a measured spec still outranks the dataset import
        blocked = await upsert_calibration(sid, _CAM, {**intr, "ref_width": 1242, "rpy_deg": [0, 0, 0],
                                                       "xyz_m": [0, 0, 1.5]}, "estimated")
        assert not blocked["stored"]
    finally:
        async with get_sessionmaker()() as db:
            await db.execute(delete(CameraCalibration).where(
                CameraCalibration.session_id == sid, CameraCalibration.cam_id == _CAM))
            await db.commit()
