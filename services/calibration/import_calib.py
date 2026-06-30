"""Explicit calib-file import (M-CAL.3c): parse the intrinsics a dataset ships (KITTI P2, nuScenes
camera_intrinsic) and store them as source=dataset, which outranks an estimate but yields to a measured
spec. The intrinsics (fx, fy, cx, cy, distortion) are exact and unambiguous, a real win over the nominal
lens defaults.

Extrinsics: nuScenes ships a sensor->ego translation, imported directly as the mount xyz; the rotation is
left on the resolved nominal/measured pose for now (decomposing an arbitrary dataset rotation into this
codebase's rpy + R_OPT2EGO convention is a separate, round-trip-verified step). So a dataset import upgrades
the intrinsics exactly and the mount translation when present, and inherits the rest.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("calibration_import")


def intrinsics_from_K(K, dist=None, model: str = "pinhole") -> dict:
    """fx, fy, cx, cy from a 3x3 camera matrix K."""
    return {"model": model, "fx": float(K[0][0]), "fy": float(K[1][1]),
            "cx": float(K[0][2]), "cy": float(K[1][2]), "dist": list(dist or [])}


def parse_kitti_calib(text: str, cam: str = "P2") -> dict:
    """Intrinsics from a KITTI calib.txt: the named projection matrix (P2 = the left colour camera) is 3x4;
    its left 3x3 is K. Raises ValueError when the line is absent or malformed."""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() == cam:
            vals = [float(v) for v in rest.split()]
            if len(vals) != 12:
                raise ValueError(f"{cam} must have 12 values, got {len(vals)}")
            p = [vals[0:4], vals[4:8], vals[8:12]]
            return intrinsics_from_K([[p[0][0], p[0][1], p[0][2]],
                                      [p[1][0], p[1][1], p[1][2]],
                                      [p[2][0], p[2][1], p[2][2]]])
    raise ValueError(f"calib line {cam} not found")


def parse_nuscenes_calib(camera_intrinsic, translation=None, dist=None) -> dict:
    """Intrinsics from a nuScenes calibrated_sensor camera_intrinsic (3x3), plus the sensor->ego translation
    as the mount xyz when given (nuScenes ego frame is x forward, y left, z up, matching this codebase)."""
    fields = intrinsics_from_K(camera_intrinsic, dist=dist)
    if translation is not None and len(translation) == 3:
        fields["xyz_m"] = [float(translation[0]), float(translation[1]), float(translation[2])]
    return fields


async def import_calibration(session_id, cam_id: str, intrinsics: dict, ref_width: int,
                             ref_height: int = 1080, source: str = "dataset") -> dict:
    """Store imported intrinsics for one camera, keeping the resolved (nominal/measured) extrinsics except
    where the import supplies a mount xyz. Respects the precedence ladder via upsert_calibration."""
    from services.calibration.resolve import nominal_calibration
    from services.calibration.store import upsert_calibration
    nom = nominal_calibration(cam_id, ref_width, ref_height)
    fields = {
        "model": intrinsics.get("model", "pinhole"),
        "fx": intrinsics["fx"], "fy": intrinsics["fy"], "cx": intrinsics["cx"], "cy": intrinsics["cy"],
        "dist": list(intrinsics.get("dist", [])), "ref_width": int(ref_width),
        "rpy_deg": list(nom.rpy_deg),                                   # rotation stays on the resolved pose
        "xyz_m": list(intrinsics.get("xyz_m", nom.xyz_m)),             # imported mount translation when present
    }
    res = await upsert_calibration(session_id, cam_id, fields, source)
    log.info("calibration.imported", session=str(session_id), cam=cam_id, source=source, stored=res["stored"])
    return res
