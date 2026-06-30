"""LiDAR-camera-IMU calibration validation, drift detection, and the 3D-annotation exclusion gate. Computes
projection and consistency residuals, runs the cloud quality checks, writes lidar_calibration_validation
rows, and flags a session whose calibration drifts or whose clouds fail quality, excluding it from 3D work
until fixed, the same contract services/calibration/report.py applies in 2D.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from db.models import LidarCalibrationValidation, PointCloud
from db.session import get_sessionmaker
from services.lidar.calib.lidar_camera import coverage_consistency, reprojection_error
from services.lidar.clean.qualitypc import check_cloud_quality
from services.lidar.ingest.store import load_cloud

log = get_logger("lidar_validate3d")


async def record_validation(session_id: uuid.UUID, pair: str, reproj_error: float | None,
                            consistency: dict, drift_flag: bool, status: str) -> uuid.UUID:
    async with get_sessionmaker()() as db:
        row = LidarCalibrationValidation(session_id=session_id, pair=pair, reproj_error=reproj_error,
                                         consistency=consistency, drift_flag=drift_flag, status=status)
        db.add(row)
        await db.flush()
        vid = row.id
        await db.commit()
    log.info("lidar.validation", session=str(session_id), pair=pair, status=status,
             reproj=reproj_error, drift=drift_flag)
    return vid


async def _baseline_residual(session_id: uuid.UUID, pair: str) -> float | None:
    async with get_sessionmaker()() as db:
        return (await db.execute(
            select(LidarCalibrationValidation.reproj_error)
            .where(LidarCalibrationValidation.session_id == session_id,
                   LidarCalibrationValidation.pair == pair,
                   LidarCalibrationValidation.reproj_error.isnot(None),
                   LidarCalibrationValidation.status != "fail")
            .order_by(LidarCalibrationValidation.created_at.desc()).limit(1))).scalar_one_or_none()


async def validate_lidar_camera(session_id: uuid.UUID, points_ego, observed_uv, cam_id: str,
                                img_w: int, img_h: int) -> dict:
    """Correspondence-based reprojection check: residual versus threshold, with drift versus the baseline.
    The precise calibration test when a target or tracked features give 3D-to-2D matches."""
    cfg = get_settings().lidar
    res = reprojection_error(points_ego, observed_uv, cam_id, img_w, img_h)
    rms = res["rms"]
    if rms is None:
        status = "fail"
    elif rms <= cfg.calib_reproj_warn_px:
        status = "pass"
    elif rms <= cfg.calib_reproj_fail_px:
        status = "warn"
    else:
        status = "fail"
    baseline = await _baseline_residual(session_id, "lidar_camera")
    drift = bool(baseline and rms and rms > baseline * cfg.calib_drift_ratio)
    if drift and status == "pass":
        status = "warn"
    vid = await record_validation(session_id, "lidar_camera", rms,
                                  {"max_px": res["max"], "n": res["n"], "baseline_px": baseline}, drift, status)
    return {"id": str(vid), "pair": "lidar_camera", "reproj_error": rms, "status": status,
            "drift_flag": drift, "baseline_px": baseline}


async def validate_session(session_id: uuid.UUID, cam_id: str = "cam_f",
                           img_w: int = 1280, img_h: int = 960) -> dict:
    """The automatic per-session check: cloud quality across every cloud plus camera coverage consistency.
    Records a validation row and returns whether the session may proceed to 3D annotation."""
    async with get_sessionmaker()() as db:
        clouds = (await db.execute(select(PointCloud).where(PointCloud.session_id == session_id)
                                   .order_by(PointCloud.ts_ns))).scalars().all()
    if not clouds:
        return {"session_id": str(session_id), "status": "fail", "reason": "no clouds in session"}

    quality = []
    worst = "pass"
    consistency: dict = {}
    for i, row in enumerate(clouds):
        cloud = load_cloud(row.cloud_uri)
        q = check_cloud_quality(cloud)
        quality.append({"cloud_id": str(row.cloud_id), "status": q["status"], "checks": q["checks"],
                        "points": q["points"], "largest_empty_wedge_deg": q["largest_empty_wedge_deg"]})
        if q["status"] == "fail":
            worst = "fail"
        if i == 0:
            consistency = coverage_consistency(cloud, cam_id, img_w, img_h)

    # a camera that sees almost none of the cloud signals a calibration mismatch
    if worst != "fail" and consistency.get("in_image_frac", 1.0) < 0.05:
        worst = "warn"
    vid = await record_validation(session_id, "lidar_camera", None,
                                  {"coverage": consistency, "quality": quality, "auto": True}, False, worst)
    return {"session_id": str(session_id), "id": str(vid), "status": worst, "clouds": len(clouds),
            "coverage": consistency, "quality": quality}


async def lidar_session_ok(session_id: uuid.UUID) -> bool:
    """The 3D-annotation gate: a session is excluded if its latest validation per pair is 'fail' or drifted.
    An un-validated session is not blocked (validation needs to have run first)."""
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(LidarCalibrationValidation)
            .where(LidarCalibrationValidation.session_id == session_id)
            .order_by(LidarCalibrationValidation.created_at.desc()))).scalars().all()
    if not rows:
        return True
    latest: dict[str, LidarCalibrationValidation] = {}
    for r in rows:
        latest.setdefault(r.pair, r)
    return all(r.status != "fail" and not r.drift_flag for r in latest.values())
