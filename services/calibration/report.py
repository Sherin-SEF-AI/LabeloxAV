"""Per-camera per-session calibration validation report (M3.0). Runs the intrinsics + FOV (and timesync
when an IMU stream is provided) checks, writes a calibration_validation row per camera plus a camera_rig
row, and a verdict. A session that fails is excluded from 3D and multi-camera work: session_calibrated()
is the downstream gate (M3.1 checks it).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select

from core.config import get_settings
from core.logging import get_logger
from db.models import CalibrationValidation, CameraRig, Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calibration.intrinsics import validate_intrinsics

log = get_logger("calib_report")


async def validate_session(session_id: UUID, imu_ts_ns: list[int] | None = None) -> dict:
    cfg = get_settings()
    maker = get_sessionmaker()
    async with maker() as db:
        sess = await db.get(DbSession, session_id)
        if sess is None:
            return {"error": "session not found"}
        cam_rows = (await db.execute(
            select(Frame.cam_id, func.max(Frame.width), func.min(Frame.ts_ns), func.max(Frame.ts_ns))
            .where(Frame.session_id == session_id).group_by(Frame.cam_id))).all()
        await db.execute(delete(CalibrationValidation).where(CalibrationValidation.session_id == session_id))

        cam_entries: dict = {}
        results: list[dict] = []
        for cam_id, width, _t0, _t1 in cam_rows:
            lens = cfg.rig.camera_lens.get(cam_id, "narrow")
            actual = cfg.rig.lenses[lens]
            intr = validate_intrinsics(actual, lens)

            time_offset, ts_ok = None, True
            if imu_ts_ns is not None:
                from services.calibration.timesync import validate_timesync

                cam_ts = (await db.execute(
                    select(Frame.ts_ns).where(Frame.session_id == session_id, Frame.cam_id == cam_id))).scalars().all()
                tsr = validate_timesync(imu_ts_ns, cam_ts_ns=list(cam_ts))
                time_offset, ts_ok = tsr["time_offset_ns"], tsr["ok"]

            status = "pass" if (intr["fov_check"]["ok"] and ts_ok) else "fail"
            db.add(CalibrationValidation(
                session_id=session_id, cam_id=cam_id, model=actual.model, reproj_error_px=intr["reproj_error_px"],
                fov_check=intr["fov_check"], time_offset_ns=time_offset, status=status))
            cam_entries[cam_id] = {"lens": lens, "model": actual.model,
                                   "intrinsics": {"fx": actual.fx, "fy": actual.fy, "cx": actual.cx, "cy": actual.cy}}
            results.append({"cam_id": cam_id, "model": actual.model, "lens": lens,
                            "reproj_error_px": intr["reproj_error_px"], "fov_check": intr["fov_check"],
                            "time_offset_ns": time_offset, "status": status})

        db.add(CameraRig(vehicle_id=sess.vehicle_id, cameras=cam_entries))
        await db.commit()

    overall = "fail" if any(r["status"] == "fail" for r in results) else "pass"
    out = {"session_id": str(session_id), "vehicle_id": sess.vehicle_id, "cameras": results, "overall": overall}
    log.info("calibration.validated", session_id=str(session_id), overall=overall, n_cameras=len(results))
    return out


async def session_calibrated(session_id: UUID) -> bool:
    """True if the session has no failing camera calibration (the 3D / multi-camera gate)."""
    maker = get_sessionmaker()
    async with maker() as db:
        n_fail = (await db.execute(
            select(func.count()).select_from(CalibrationValidation)
            .where(CalibrationValidation.session_id == session_id, CalibrationValidation.status == "fail"))).scalar_one()
        n_total = (await db.execute(
            select(func.count()).select_from(CalibrationValidation)
            .where(CalibrationValidation.session_id == session_id))).scalar_one()
    return n_total > 0 and n_fail == 0
