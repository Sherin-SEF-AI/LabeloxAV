"""Calibration validation endpoints (M3.0): validate a session's cameras, list the per-camera verdicts,
and the session-level pass/fail that gates 3D + multi-camera work."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CalibrationValidation, Frame
from db.models import Session as DbSession
from services.api.deps import db_session

router = APIRouter()


@router.post("/calibration/validate")
async def validate(session_id: str):
    from services.calibration.report import validate_session

    return await validate_session(UUID(session_id))


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
