"""M-CAL.3a: writing real calibration into the store. spec_to_fields turns a human rig spec (focal or FOV,
mount height, pitch) into stored fields; upsert_calibration respects source precedence so a measured spec is
never downgraded by an estimate; and resolve_calibration then reads the stored measured row, scaled to the
image, instead of the nominal default."""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from db.models import CameraCalibration
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calibration.resolve import resolve_calibration
from services.calibration.store import spec_to_fields, upsert_calibration

_CAM = "cam_test_mcal3"


def test_spec_fov_to_focal():
    f = spec_to_fields(1920, 1080, {"hfov_deg": 37.0, "height_m": 1.6, "pitch_deg": -2.0})
    assert abs(f["fx"] - 2870.0) < 6.0          # 1920 / (2 tan(18.5deg))
    assert f["fy"] == f["fx"]
    assert f["cx"] == 960.0 and f["cy"] == 540.0
    assert f["xyz_m"] == [0.0, 0.0, 1.6]
    assert f["rpy_deg"] == [0.0, -2.0, 0.0]


def test_spec_explicit_fx_and_principal_point():
    f = spec_to_fields(1280, 960, {"fx": 1000.0, "cx": 640.0})
    assert f["fx"] == 1000.0 and f["fy"] == 1000.0
    assert f["cx"] == 640.0 and f["cy"] == 480.0   # cy defaults to image centre


def test_spec_requires_a_focal_source():
    with pytest.raises(ValueError):
        spec_to_fields(1920, 1080, {"height_m": 1.5})


async def test_upsert_precedence_then_resolve_reads_measured():
    async with get_sessionmaker()() as db:
        sid = (await db.execute(select(DbSession.session_id).limit(1))).scalar()
    assert sid is not None, "need at least one session in the corpus"
    try:
        est = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 2000.0, "height_m": 1.4}),
                                       "estimated")
        assert est["stored"]
        meas = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 2870.0, "height_m": 1.6}),
                                        "measured")
        assert meas["stored"]
        # an estimate must not downgrade the measured row
        down = await upsert_calibration(sid, _CAM, spec_to_fields(1920, 1080, {"fx": 1500.0}), "estimated")
        assert not down["stored"] and "higher-trust" in down["reason"]

        c = await resolve_calibration(sid, _CAM, 960, 540)   # half the reference width
        assert c.source == "measured"
        assert abs(c.fx - 1435.0) < 1.0                      # 2870 * (960 / 1920)
        assert abs(c.xyz_m[2] - 1.6) < 1e-6
    finally:
        async with get_sessionmaker()() as db:
            await db.execute(delete(CameraCalibration).where(
                CameraCalibration.session_id == sid, CameraCalibration.cam_id == _CAM))
            await db.commit()
