"""CALYX calibration-monitor endpoints: record an estimated extrinsic drift for a session, read a session's
calibration state, and read a rig's drift-over-time timeline. The reprojection-residual and epipolar views
reuse the existing /calibration router; CALYX adds the SE(3) drift-delta and the rig history the data engine
needs. Mounted under /api; the platform registry maps /calyx (and legacy /calibration) here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CalibrationValidation
from db.models import Session as DbSession
from services.api.deps import db_session
from services.calyx.run import record_drift, rig_history

router = APIRouter()


class DriftIn(BaseModel):
    cam_id: str
    cam_points: list[list[float]]        # camera-observed 3D positions over a window
    inertial_points: list[list[float]]   # corresponding IMU/GNSS-reported positions


@router.post("/calyx/session/{session_id}/drift")
async def record(session_id: uuid.UUID, payload: DriftIn, db: AsyncSession = Depends(db_session)):
    """Estimate and persist the extrinsic drift for one camera on a session from corresponding camera vs
    inertial 3D positions."""
    if await db.get(DbSession, session_id) is None:
        raise HTTPException(404, "session not found")
    return await record_drift(session_id, payload.cam_id, payload.cam_points, payload.inertial_points)


@router.get("/calyx/session/{session_id}")
async def session_state(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """The calibration state for a session: the per-camera drift deltas and severities, latest first."""
    rows = (await db.execute(
        select(CalibrationValidation).where(CalibrationValidation.session_id == session_id,
                                            CalibrationValidation.severity.isnot(None))
        .order_by(CalibrationValidation.created_at.desc()))).scalars().all()
    return {"session_id": str(session_id), "cameras": [
        {"cam_id": r.cam_id, "severity": r.severity, "status": r.status,
         "drift": r.drift_delta, "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows]}


@router.get("/calyx/rig/{vehicle_id}/history")
async def rig(vehicle_id: str):
    """The rig-level drift timeline, so a slowly loosening mount is caught before it blocks."""
    return await rig_history(vehicle_id)
