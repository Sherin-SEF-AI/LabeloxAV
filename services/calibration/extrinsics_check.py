"""Enable the dormant extrinsic validation (M-CAL.4). The epipolar (Sampson) math in extrinsics.py was
unit-only because nothing supplied real correspondences and the relative pose. This wires it to the live
data: the same object seen in two rig cameras (rig_track_id / cross_cam_links) gives matched image points,
the resolved calibration gives each camera's intrinsics and ego pose, and the relative pose between them
follows. A low Sampson residual means the two cameras' extrinsics are consistent with what they observe.

Single-camera sessions (most of the dashcam fleet) have no overlapping pair, so the check reports that
honestly rather than inventing a number; it runs where a real multi-camera rig exists.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger
from services.calibration.extrinsics import epipolar_residual

log = get_logger("calibration_extrinsics_check")


def relative_pose(cal_a, cal_b) -> tuple[np.ndarray, np.ndarray]:
    """The relative pose (R, t) mapping camera-a optical to camera-b optical, x_b = R x_a + t, from two
    resolved calibrations. Calibration.R() is the row-convention ego->optical matrix, so its transpose is the
    column-convention ego->optical rotation used here."""
    ra = cal_a.R().T.astype(np.float64)        # column-convention ego -> cam_a
    rb = cal_b.R().T.astype(np.float64)         # column-convention ego -> cam_b
    r_rel = rb @ ra.T
    t_rel = rb @ (cal_a.t().astype(np.float64) - cal_b.t().astype(np.float64))
    return r_rel, t_rel


def epipolar_consistency(pts_a, pts_b, cal_a, cal_b) -> dict:
    """Sampson residual (px) of matched points across two cameras under their resolved relative pose."""
    r_rel, t_rel = relative_pose(cal_a, cal_b)
    return epipolar_residual(pts_a, pts_b, r_rel, t_rel, cal_a.K(), cal_b.K())


async def check_session_extrinsics(session_id, write: bool = True) -> dict:
    """Cross-camera extrinsic consistency for a session: gather the matched centres of objects seen in two
    cameras (via rig_track_id), resolve each camera's calibration, and score the Sampson residual per camera
    pair. Writes the result onto each camera's CalibrationValidation.extrinsic_consistency when present."""
    from collections import defaultdict

    from sqlalchemy import select

    from core.schemas import BBox
    from db.models import CalibrationValidation, Frame, Object
    from db.session import get_sessionmaker
    from services.calibration.resolve import resolve_calibration

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.rig_track_id, Frame.cam_id, Frame.width, Frame.height, Object.bbox)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, Object.rig_track_id.isnot(None)))).all()

    # rig_track_id -> {cam_id: (center_uv, w, h)}: the same physical object across cameras at one instant
    by_track: dict = defaultdict(dict)
    for rig_id, cam_id, w, h, bbox in rows:
        cx, cy = BBox.from_list(list(bbox)).center
        by_track[rig_id][cam_id] = ((float(cx), float(cy)), int(w), int(h))

    pair_pts: dict = defaultdict(lambda: ([], []))
    pair_dims: dict = {}
    for cams in by_track.values():
        ids = sorted(cams)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pair_pts[(a, b)][0].append(cams[a][0])
                pair_pts[(a, b)][1].append(cams[b][0])
                pair_dims[(a, b)] = (cams[a][1], cams[a][2], cams[b][1], cams[b][2])

    if not pair_pts:
        return {"session_id": str(session_id), "checked": False, "reason": "no overlapping cameras"}

    pairs = []
    for (a, b), (pa, pb) in pair_pts.items():
        if len(pa) < 8:                          # too few correspondences for a stable residual
            continue
        wa, ha, wb, hb = pair_dims[(a, b)]
        cal_a = await resolve_calibration(session_id, a, wa, ha)
        cal_b = await resolve_calibration(session_id, b, wb, hb)
        res = epipolar_consistency(pa, pb, cal_a, cal_b)
        pairs.append({"cam_a": a, "cam_b": b, "mean_sampson_px": round(res["mean_sampson_px"], 2)
                      if res["mean_sampson_px"] is not None else None, "n": res["n"],
                      "calib_sources": sorted({cal_a.source, cal_b.source})})

    if write and pairs:
        async with get_sessionmaker()() as db:
            for p in pairs:
                for cam in (p["cam_a"], p["cam_b"]):
                    row = (await db.execute(select(CalibrationValidation).where(
                        CalibrationValidation.session_id == session_id,
                        CalibrationValidation.cam_id == cam))).scalar_one_or_none()
                    if row is not None:
                        row.extrinsic_consistency = {"mean_sampson_px": p["mean_sampson_px"],
                                                     "pair": f"{p['cam_a']}|{p['cam_b']}", "n": p["n"]}
            await db.commit()

    worst = max((p["mean_sampson_px"] for p in pairs if p["mean_sampson_px"] is not None), default=None)
    log.info("calibration.extrinsics_checked", session=str(session_id), pairs=len(pairs), worst=worst)
    return {"session_id": str(session_id), "checked": True, "pairs": pairs, "worst_sampson_px": worst}
