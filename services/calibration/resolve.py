"""The calibration seam (M-CAL.1). Every 3D consumer (projection, pseudo-LiDAR lift, the 2D-3D link) reads a
single resolved Calibration per (session, camera) instead of reaching into the config rig defaults. When a
session has stored real calibration it is used; otherwise the nominal rig calibration is returned, tagged so
a cuboid's trust follows its calibration source. This is the foundation the whole vision-first 3D plane
rides on: replace one resolver with real per-vehicle calibration and every metric-3D number improves.

Ego frame is x forward, y left, z up; camera optical frame is x right, y down, z forward. Extrinsics are the
full 6-DOF camera->ego mount pose (roll, pitch, yaw and the mount xyz), generalizing the legacy per-camera
yaw plus a single mount height. With pitch=roll=0 and xyz=(0,0,height) the resolved matrices reproduce the
legacy projection exactly, so the nominal path is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.config import get_settings

# optical (x right, y down, z forward) -> ego (x forward, y left, z up); shared with the pseudo-LiDAR lift
_R_OPT2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)

# how far to trust each calibration source, in [0, 1]; flows onto the cuboid as a quality signal
SOURCE_QUALITY = {"measured": 1.0, "dataset": 0.9, "estimated": 0.6, "nominal": 0.3}


@dataclass
class Calibration:
    """A resolved camera calibration at a specific image size. fx/fy/cx/cy are already scaled to (img_w,
    img_h). rpy_deg and xyz_m are the camera->ego mount pose. source/quality carry the provenance."""

    cam_id: str
    model: str                                   # pinhole | fisheye
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=list)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)   # roll, pitch, yaw (ego -> camera mount)
    xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)     # camera mount position in the ego frame
    source: str = "nominal"
    quality: float = 0.3

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def R(self) -> np.ndarray:
        """The 3x3 mapping an ego-shifted point (row vector) to the camera optical frame: cam = (ego - t) @ R.
        With pitch=roll=0 this is exactly the legacy Rz(-yaw) @ R_OPT2EGO."""
        roll, pitch, yaw = (math.radians(a) for a in self.rpy_deg)
        cz, sz = math.cos(-yaw), math.sin(-yaw)
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        cp, sp = math.cos(-pitch), math.sin(-pitch)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        cr, sr = math.cos(-roll), math.sin(-roll)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return (rz @ ry @ rx @ _R_OPT2EGO).astype(np.float32)

    def t(self) -> np.ndarray:
        return np.array(self.xyz_m, dtype=np.float32)


def nominal_calibration(cam_id: str, img_w: int, img_h: int) -> Calibration:
    """The rig-default calibration from config: lens intrinsics scaled to the image (principal point at the
    image centre), per-camera yaw, and the global mount height. This reproduces the legacy projection."""
    cfg = get_settings()
    lens_name = cfg.rig.camera_lens.get(cam_id, "narrow")
    k = cfg.rig.lenses[lens_name]
    scale = img_w / cfg.rig.ref_width
    return Calibration(
        cam_id=cam_id, model=k.model, fx=k.fx * scale, fy=k.fy * scale, cx=img_w / 2.0, cy=img_h / 2.0,
        dist=list(k.dist), rpy_deg=(0.0, 0.0, cfg.rig.camera_yaw_deg.get(cam_id, 0.0)),
        xyz_m=(0.0, 0.0, cfg.spatial.camera_height_m), source="nominal", quality=SOURCE_QUALITY["nominal"],
    )


def calibration_from_row(row, img_w: int, img_h: int) -> Calibration:
    """Build a Calibration from a stored camera_calibration row, scaling the intrinsics from ref_width to the
    actual image. The real principal point is scaled too, not forced to the image centre."""
    scale = img_w / float(row.ref_width or img_w)
    rpy = tuple(row.rpy_deg or (0.0, 0.0, 0.0))
    xyz = tuple(row.xyz_m or (0.0, 0.0, 0.0))
    return Calibration(
        cam_id=row.cam_id, model=row.model, fx=row.fx * scale, fy=row.fy * scale,
        cx=row.cx * scale, cy=row.cy * scale, dist=list(row.dist or []),
        rpy_deg=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        xyz_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        source=row.source, quality=float(row.quality),
    )


async def resolve_calibration(session_id, cam_id: str, img_w: int, img_h: int) -> Calibration:
    """The stored per-session calibration for this camera, or the nominal rig default when none exists."""
    from sqlalchemy import select

    from db.models import CameraCalibration
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        row = (await db.execute(select(CameraCalibration).where(
            CameraCalibration.session_id == session_id, CameraCalibration.cam_id == cam_id))).scalar_one_or_none()
    if row is None:
        return nominal_calibration(cam_id, img_w, img_h)
    return calibration_from_row(row, img_w, img_h)
