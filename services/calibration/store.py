"""Write real calibration into the camera_calibration store the resolver reads (M-CAL.3). One row per
(session, camera); a write respects source precedence so a measured spec is never clobbered by an estimate:
measured > dataset > estimated > nominal. nominal is never stored (it is the fallback). All three ingestion
paths (per-vehicle spec, monocular estimation, calib-file import) write through upsert_calibration.

A human-facing spec (focal length or horizontal FOV, mount height, pitch) is turned into the stored fields
by spec_to_fields: fx from the FOV when no focal is given, the principal point at the image centre unless
provided, and the extrinsics as roll/pitch/yaw plus the mount xyz.
"""

from __future__ import annotations

import math

from sqlalchemy import select

from core.logging import get_logger
from db.models import CameraCalibration
from db.session import get_sessionmaker
from services.calibration.resolve import SOURCE_QUALITY

log = get_logger("calibration_store")

# trust ordering: a write never downgrades a higher-trust source
_RANK = {"nominal": 0, "estimated": 1, "dataset": 2, "measured": 3}


def spec_to_fields(ref_width: int, ref_height: int, spec: dict) -> dict:
    """Turn a human calibration spec into stored camera_calibration fields. spec keys (all optional except a
    focal source): fx | hfov_deg, fy, cx, cy, dist, model, yaw_deg, pitch_deg, roll_deg, height_m, x_m, y_m."""
    if "fx" in spec and spec["fx"]:
        fx = float(spec["fx"])
    elif spec.get("hfov_deg"):
        fx = ref_width / (2.0 * math.tan(math.radians(float(spec["hfov_deg"])) / 2.0))
    else:
        raise ValueError("calibration spec needs fx or hfov_deg")
    fy = float(spec.get("fy", fx))
    return {
        "model": spec.get("model", "pinhole"),
        "fx": fx, "fy": fy,
        "cx": float(spec.get("cx", ref_width / 2.0)),
        "cy": float(spec.get("cy", ref_height / 2.0)),
        "dist": list(spec.get("dist", [])),
        "ref_width": int(ref_width),
        "rpy_deg": [float(spec.get("roll_deg", 0.0)), float(spec.get("pitch_deg", 0.0)),
                    float(spec.get("yaw_deg", 0.0))],
        "xyz_m": [float(spec.get("x_m", 0.0)), float(spec.get("y_m", 0.0)), float(spec.get("height_m", 1.5))],
    }


async def upsert_calibration(session_id, cam_id: str, fields: dict, source: str,
                             quality: float | None = None) -> dict:
    """Store or overwrite one camera's calibration, never downgrading a higher-trust source. fields is the
    output of spec_to_fields (or an importer/estimator). Returns whether it was stored or skipped."""
    rank = _RANK.get(source, 0)
    quality = quality if quality is not None else SOURCE_QUALITY.get(source, 0.3)
    async with get_sessionmaker()() as db:
        row = (await db.execute(select(CameraCalibration).where(
            CameraCalibration.session_id == session_id, CameraCalibration.cam_id == cam_id))).scalar_one_or_none()
        if row is not None and _RANK.get(row.source, 0) > rank:
            return {"stored": False, "reason": "existing higher-trust source", "source": row.source}
        if row is None:
            row = CameraCalibration(session_id=session_id, cam_id=cam_id)
            db.add(row)
        row.model = fields["model"]
        row.fx, row.fy, row.cx, row.cy = fields["fx"], fields["fy"], fields["cx"], fields["cy"]
        row.dist = fields["dist"]
        row.ref_width = fields["ref_width"]
        row.rpy_deg = fields["rpy_deg"]
        row.xyz_m = fields["xyz_m"]
        row.source = source
        row.quality = quality
        await db.commit()
    log.info("calibration.stored", session=str(session_id), cam=cam_id, source=source)
    return {"stored": True, "source": source, "cam_id": cam_id}


async def set_session_calibration(session_id, cam_specs: dict, source: str = "measured",
                                  ref_width: int = 1920, ref_height: int = 1080) -> dict:
    """Set calibration for one session from a {cam_id: spec} map (the per-vehicle rig spec path)."""
    results = {}
    for cam_id, spec in cam_specs.items():
        fields = spec_to_fields(spec.get("ref_width", ref_width), spec.get("ref_height", ref_height), spec)
        results[cam_id] = await upsert_calibration(session_id, cam_id, fields, source)
    return {"session_id": str(session_id), "source": source, "cameras": results}


async def apply_vehicle_calibration(vehicle_id: str, cam_specs: dict, source: str = "measured") -> dict:
    """Apply one rig spec to every session of a vehicle: a known fleet rig is entered once and stamped onto
    all its captures (existing and future sessions re-resolve through the same store)."""
    from db.models import Session as DbSession
    async with get_sessionmaker()() as db:
        sids = (await db.execute(
            select(DbSession.session_id).where(DbSession.vehicle_id == vehicle_id))).scalars().all()
    applied = 0
    for sid in sids:
        await set_session_calibration(sid, cam_specs, source)
        applied += 1
    log.info("calibration.vehicle_applied", vehicle=vehicle_id, sessions=applied, source=source)
    return {"vehicle_id": vehicle_id, "sessions": applied, "source": source}
