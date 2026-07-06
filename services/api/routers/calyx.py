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

from db.models import CalibrationOverride, CalibrationValidation
from db.models import Session as DbSession
from services.api.deps import db_session
from services.calyx.consensus import fuse_calibrations
from services.calyx.recover import record_override
from services.calyx.run import record_drift, rig_history
from services.calyx.targetless import calibrate_targetless

router = APIRouter()


class RecoverIn(BaseModel):
    cam_id: str
    nominal: list[list[float]]        # 4x4 nominal extrinsic
    drift_delta: list[list[float]]    # 4x4 SE(3) drift from CALYX
    residual_px: float
    n_correspondences: int
    severity: str = "drift_detected"


@router.post("/calyx/session/{session_id}/recover")
async def recover_session(session_id: uuid.UUID, payload: RecoverIn, db: AsyncSession = Depends(db_session)):
    """Self-calibrate a recoverable session: compute the correction that undoes the measured drift and store
    it as a versioned override, so the session becomes usable instead of quarantined."""
    if await db.get(DbSession, session_id) is None:
        raise HTTPException(404, "session not found")
    return await record_override(session_id, payload.cam_id, payload.nominal, payload.drift_delta,
                                 payload.residual_px, payload.n_correspondences, payload.severity)


class TargetlessIn(BaseModel):
    vp1: list[float]
    vp2: list[float]
    horizon_y: float
    cx: float
    cy: float


@router.post("/calyx/targetless")
async def targetless(payload: TargetlessIn):
    """Recover focal length and pitch from natural-scene cues (orthogonal vanishing points + horizon), so a
    field session calibrates from the scene it recorded, no target needed."""
    return calibrate_targetless(payload.vp1, payload.vp2, payload.horizon_y, payload.cx, payload.cy)


@router.get("/calyx/rig/{vehicle_id}/consensus")
async def rig_consensus(vehicle_id: str, db: AsyncSession = Depends(db_session)):
    """Fuse the vehicle's per-session calibration overrides into a stronger rig prior."""
    rows = (await db.execute(
        select(CalibrationOverride)
        .join(DbSession, CalibrationOverride.session_id == DbSession.session_id)
        .where(DbSession.vehicle_id == vehicle_id).limit(500))).scalars().all()
    calibs = [{"rpy_deg": (r.corrected or {}).get("rpy_deg"), "xyz_m": (r.corrected or {}).get("xyz_m"),
               "confidence": r.confidence} for r in rows]
    return {"vehicle_id": vehicle_id, "n_overrides": len(calibs), "prior": fuse_calibrations(calibs)}


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
