"""CALYX persistence: record an estimated drift onto the session's calibration_validation row (drift_delta +
severity) and read the rig-level drift history, so a slowly loosening mount is caught over many sessions
before it becomes catastrophic. The estimation core is drift.py; this wires it to the spine."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from db.models import CalibrationValidation
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.calyx.drift import estimate_extrinsic_drift

log = get_logger("calyx.run")

_STATUS = {"ok": "pass", "drift_detected": "warn", "block": "fail"}


async def record_drift(session_id: UUID, cam_id: str, cam_points, inertial_points) -> dict:
    """Estimate the extrinsic drift for one camera on one session and persist it as a calibration_validation
    row carrying the SE(3) drift delta and severity. Returns the estimate."""
    cfg = get_settings().calyx
    est = estimate_extrinsic_drift(np.asarray(cam_points), np.asarray(inertial_points), cfg)
    async with get_sessionmaker()() as db:
        row = CalibrationValidation(
            session_id=session_id, cam_id=cam_id, model="pinhole",
            drift_delta={"delta": est["delta"], "magnitude": est["magnitude"], "residual": est["residual"],
                         "n": est["n"], "confidence": est["confidence"]},
            severity=est["severity"], status=_STATUS[est["severity"]],
        )
        db.add(row)
        await db.commit()
    log.info("calyx.drift", session=str(session_id), cam=cam_id, severity=est["severity"],
             rot=est["magnitude"]["rotation_deg"], trans=est["magnitude"]["translation_m"])
    return est


async def rig_history(vehicle_id: str, limit: int = 200) -> dict:
    """The drift-over-time timeline for a rig: per session, the worst drift magnitude and severity across its
    cameras, oldest to newest, so a monotonic climb (a loosening mount) is visible before it blocks."""
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(CalibrationValidation, DbSession.created_at, DbSession.session_id)
            .join(DbSession, CalibrationValidation.session_id == DbSession.session_id)
            .where(DbSession.vehicle_id == vehicle_id, CalibrationValidation.severity.isnot(None))
            .order_by(DbSession.created_at.asc()).limit(limit))).all()
    by_session: dict = {}
    for cv, created, sid in rows:
        mag = (cv.drift_delta or {}).get("magnitude", {}) if cv.drift_delta else {}
        rot = float(mag.get("rotation_deg", 0.0))
        rank = {"ok": 0, "drift_detected": 1, "block": 2}
        cur = by_session.get(str(sid))
        if cur is None or rank.get(cv.severity, 0) > rank.get(cur["severity"], 0):
            by_session[str(sid)] = {
                "session_id": str(sid), "created_at": created.isoformat() if created else None,
                "severity": cv.severity, "cam_id": cv.cam_id,
                "rotation_deg": rot, "translation_m": float(mag.get("translation_m", 0.0)),
            }
    timeline = sorted(by_session.values(), key=lambda x: x["created_at"] or "")
    return {"vehicle_id": vehicle_id, "n_sessions": len(timeline), "timeline": timeline}
