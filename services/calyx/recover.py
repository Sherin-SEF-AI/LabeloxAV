"""CALYX self-calibration recovery (M11). A session with recoverable drift (drift_detected, not block) does
not have to be thrown away: CALYX estimates the correction that undoes the measured SE(3) drift and stores it
as a versioned calibration override on the session, so the session becomes usable while the raw calibration is
never mutated. The override carries its confidence and provenance so ORACLYX and the workspace can weight how
much to trust the recovered geometry.

correct_extrinsic() is pure over the geometry, so recovery is deterministic and testable."""

from __future__ import annotations

from uuid import UUID

import numpy as np

from core.geometry import se3_compose, se3_inverse, se3_magnitude
from core.logging import get_logger
from db.models import CalibrationOverride
from db.session import get_sessionmaker
from services.calyx.uncertainty import calibration_confidence

log = get_logger("calyx.recover")


def correct_extrinsic(nominal: np.ndarray, drift_delta: np.ndarray) -> np.ndarray:
    """The corrected extrinsic that undoes the measured drift. drift_delta is the SE(3) that maps the
    camera-observed frame onto the inertial frame (from CALYX estimate_extrinsic_drift); applying its inverse
    to the nominal extrinsic removes the drift."""
    return se3_compose(se3_inverse(np.asarray(drift_delta, dtype=np.float64)), np.asarray(nominal, dtype=np.float64))


def is_recoverable(severity: str, confidence: float, min_confidence: float | None = None) -> bool:
    """A session is recoverable when the drift is flagged but not blocking, and the correction is confident
    enough to trust. A block-severity drift is too large to recover safely."""
    floor = min_confidence if min_confidence is not None else 0.4
    return severity == "drift_detected" and confidence >= floor


def recover(nominal: np.ndarray, drift_delta: np.ndarray, residual_px: float, n_corr: int,
            severity: str) -> dict:
    """Compute the recovery for one camera: the corrected extrinsic, its confidence, and whether it should be
    applied. Residual is expected to shrink after correction (the whole point)."""
    corrected = correct_extrinsic(nominal, drift_delta)
    residual_delta = se3_magnitude(np.asarray(drift_delta, dtype=np.float64))
    conf = calibration_confidence(residual_px, n_corr)
    apply = is_recoverable(severity, conf)
    return {"corrected": corrected.tolist(), "confidence": round(conf, 4), "apply": apply,
            "residual_removed": residual_delta, "severity": severity}


def _extrinsic_dict(corrected: np.ndarray) -> dict:
    from scipy.spatial.transform import Rotation
    c = np.asarray(corrected, dtype=np.float64)
    rpy = Rotation.from_matrix(c[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return {"rpy_deg": [round(v, 4) for v in rpy], "xyz_m": [round(v, 4) for v in c[:3, 3]]}


async def record_override(session_id: UUID, cam_id: str, nominal, drift_delta, residual_px: float,
                          n_corr: int, severity: str) -> dict:
    """Compute and persist a self-cal override when the session is recoverable."""
    import numpy as _np
    res = recover(_np.asarray(nominal), _np.asarray(drift_delta), residual_px, n_corr, severity)
    async with get_sessionmaker()() as db:
        prior = (await _latest_version(db, session_id, cam_id))
        row = CalibrationOverride(
            session_id=session_id, cam_id=cam_id, source="self_cal",
            corrected=_extrinsic_dict(res["corrected"]), confidence=res["confidence"],
            provenance={"method": "self_cal_from_drift", "residual_px": residual_px, "n": n_corr,
                        "residual_removed": res["residual_removed"], "severity": severity},
            version=prior + 1, applied=res["apply"])
        db.add(row)
        await db.commit()
        oid = str(row.override_id)
    log.info("calyx.recover", session=str(session_id), cam=cam_id, applied=res["apply"],
             confidence=res["confidence"])
    return {"override_id": oid, "applied": res["apply"], "confidence": res["confidence"],
            "corrected": _extrinsic_dict(res["corrected"])}


async def _latest_version(db, session_id, cam_id) -> int:
    from sqlalchemy import func, select
    v = (await db.execute(
        select(func.max(CalibrationOverride.version)).where(
            CalibrationOverride.session_id == session_id, CalibrationOverride.cam_id == cam_id))).scalar()
    return int(v or 0)
