"""Calibration validation endpoints (M3.0): validate a session's cameras, list the per-camera verdicts,
and the session-level pass/fail that gates 3D + multi-camera work."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CalibrationValidation, Frame
from db.models import Session as DbSession
from services.api.deps import db_session

router = APIRouter()


class SetCalibIn(BaseModel):
    # cam_specs: {cam_id: {fx|hfov_deg, fy?, cx?, cy?, dist?, model?, yaw_deg?, pitch_deg?, roll_deg?,
    # height_m?, x_m?, y_m?, ref_width?, ref_height?}}. source in measured|dataset|estimated.
    cam_specs: dict
    source: str = "measured"
    ref_width: int = 1920
    ref_height: int = 1080


class ImportCalibIn(BaseModel):
    cam_id: str
    format: str                               # kitti | nuscenes
    ref_width: int = 1920
    ref_height: int = 1080
    calib_text: str | None = None             # kitti calib.txt contents
    camera_intrinsic: list | None = None      # nuscenes 3x3 K
    translation: list | None = None           # nuscenes sensor->ego xyz
    dist: list | None = None


@router.post("/calibration/validate")
async def validate(session_id: str):
    from services.calibration.report import validate_session

    return await validate_session(UUID(session_id))


@router.post("/calibration/vehicle/{vehicle_id}/calibrate")
async def calibrate_vehicle(vehicle_id: str, body: SetCalibIn):
    """Stamp one known rig spec onto every session of a vehicle (the per-vehicle real-calibration path)."""
    from services.calibration.store import apply_vehicle_calibration
    specs = {c: {**s, "ref_width": s.get("ref_width", body.ref_width),
                 "ref_height": s.get("ref_height", body.ref_height)} for c, s in body.cam_specs.items()}
    return await apply_vehicle_calibration(vehicle_id, specs, body.source)


@router.post("/calibration/{session_id}/calibrate")
async def calibrate_session(session_id: UUID, body: SetCalibIn):
    """Set real calibration for one session's cameras from a rig spec (focal/FOV, mount height, pitch)."""
    from services.calibration.store import set_session_calibration
    return await set_session_calibration(session_id, body.cam_specs, body.source, body.ref_width, body.ref_height)


@router.post("/calibration/{session_id}/estimate")
async def estimate_session(session_id: UUID):
    """Monocular estimation: recover the camera pitch from road lines (and focal from EXIF when present) for
    a session's cameras, stored as source=estimated. Upgrades a session off nominal without a calib file."""
    from services.calibration.estimate import estimate_session_calibration
    return await estimate_session_calibration(session_id)


@router.post("/calibration/{session_id}/import")
async def import_calib(session_id: UUID, body: ImportCalibIn):
    """Import dataset intrinsics (KITTI P2 or nuScenes camera_intrinsic) for one camera, stored as
    source=dataset. Exact focal and principal point, a real win over the nominal lens."""
    from fastapi import HTTPException

    from services.calibration.import_calib import (
        import_calibration,
        parse_kitti_calib,
        parse_nuscenes_calib,
    )
    if body.format == "kitti":
        if not body.calib_text:
            raise HTTPException(422, "kitti import needs calib_text")
        intr = parse_kitti_calib(body.calib_text)
    elif body.format == "nuscenes":
        if not body.camera_intrinsic:
            raise HTTPException(422, "nuscenes import needs camera_intrinsic")
        intr = parse_nuscenes_calib(body.camera_intrinsic, body.translation, body.dist)
    else:
        raise HTTPException(422, f"unknown calib format {body.format}")
    return await import_calibration(session_id, body.cam_id, intr, body.ref_width, body.ref_height)


@router.get("/calibration/sessions")
async def list_sessions(db: AsyncSession = Depends(db_session)):
    """Sessions that have been validated, with their overall verdict (the report-viewer index)."""
    rows = (await db.execute(
        select(CalibrationValidation.session_id, CalibrationValidation.status, DbSession.vehicle_id)
        .join(DbSession, DbSession.session_id == CalibrationValidation.session_id))).all()
    by: dict = {}
    for sid, status, veh in rows:
        s = by.setdefault(str(sid), {"session_id": str(sid), "vehicle_id": veh, "cameras": 0, "fail": 0})
        s["cameras"] += 1
        if status == "fail":
            s["fail"] += 1
    out = []
    for s in by.values():
        s["overall"] = "fail" if s["fail"] else "pass"
        out.append(s)
    return out


@router.get("/calibration/{session_id}/resolved")
async def resolved(session_id: UUID):
    """Every camera's resolved calibration (stored real or nominal) plus the session trust level. Drives the
    ingestion UI and the calibration-trust surface."""
    from services.calibration.resolve import resolved_session_calibration
    return await resolved_session_calibration(session_id)


@router.post("/calibration/{session_id}/extrinsics")
async def check_extrinsics(session_id: UUID):
    """Cross-camera extrinsic consistency: score the epipolar (Sampson) residual of objects seen in two rig
    cameras under their resolved calibration. Single-camera sessions report no overlapping pair."""
    from services.calibration.extrinsics_check import check_session_extrinsics
    return await check_session_extrinsics(session_id)


@router.get("/calibration/{session_id}")
async def get_validation(session_id: UUID, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(
        select(CalibrationValidation).where(CalibrationValidation.session_id == session_id)
        .order_by(CalibrationValidation.cam_id))).scalars().all()
    cams = (await db.execute(
        select(Frame.cam_id).where(Frame.session_id == session_id).distinct())).scalars().all()
    return {
        "session_id": str(session_id), "cameras_in_session": list(cams),
        "validations": [{"cam_id": r.cam_id, "model": r.model, "reproj_error_px": r.reproj_error_px,
                         "fov_check": r.fov_check, "time_offset_ns": r.time_offset_ns, "status": r.status}
                        for r in rows],
        "overall": "fail" if any(r.status == "fail" for r in rows) else ("pass" if rows else "unvalidated"),
    }
