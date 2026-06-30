"""Monocular calibration estimation (M-CAL.3b). For the uncalibrated corpus (dashcam, BDD, Mapillary, IDD),
recover what a single road image honestly affords: a focal length from EXIF when the metadata survives, and
the camera pitch from the forward vanishing point of road and lane lines (the horizon position). Stored as
source=estimated, which sits below dataset and measured in the precedence ladder and never overwrites them.

This is a geometric guess, not a measurement: a real pitch (better than the nominal 0) and, only when EXIF
is present, a real focal. When no lines converge or no EXIF focal exists, the missing field falls back to
the nominal lens, so the result is always at least as good as nominal and tagged honestly.
"""

from __future__ import annotations

import math

import numpy as np

from core.logging import get_logger

log = get_logger("calibration_estimate")


def focal_from_exif(exif: dict | None, img_w: int) -> float | None:
    """fx from a 35mm-equivalent focal length (the field that survives most transcodes), or None."""
    f35 = (exif or {}).get("FocalLengthIn35mmFilm") or (exif or {}).get("focal_35mm")
    if not f35:
        return None
    try:
        return float(f35) / 36.0 * float(img_w)   # 36mm = full-frame sensor width
    except (TypeError, ValueError):
        return None


def vanishing_point(lines: list) -> tuple[float, float] | None:
    """Least-squares intersection (u, v) of 2+ line segments [(x1, y1, x2, y2)], the vanishing point of a set
    of parallel world lines. None when fewer than two usable lines."""
    rows_a, rows_b = [], []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        a, b = dy, -dx                       # the line's normal
        n = math.hypot(a, b)
        if n < 1e-6:
            continue
        a, b = a / n, b / n
        rows_a.append([a, b])
        rows_b.append(a * x1 + b * y1)       # a*u + b*v = a*x1 + b*y1
    if len(rows_a) < 2:
        return None
    sol, *_ = np.linalg.lstsq(np.asarray(rows_a), np.asarray(rows_b), rcond=None)
    return float(sol[0]), float(sol[1])


def pitch_from_vp(v_vp: float, fy: float, cy: float) -> float:
    """Camera pitch in radians (down positive) from the forward vanishing point's vertical pixel: a VP above
    the principal point means the optical axis is pitched down toward the road."""
    return math.atan2(cy - v_vp, fy)


def _road_lines(image_bgr: np.ndarray) -> list:
    """Oblique long edges in the lower half of the frame: candidate road and lane lines for the forward VP."""
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    h = edges.shape[0]
    edges[: h // 2] = 0                       # drop the sky; the road lines are below the horizon
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(20, h // 6), maxLineGap=20)
    if segs is None:
        return []
    lines = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
        if 20 < ang < 80:                     # oblique (converging) lines, not horizontal or vertical
            lines.append((float(x1), float(y1), float(x2), float(y2)))
    return lines


def estimate_frame(image_bgr: np.ndarray, cx: float, cy: float, fy: float) -> dict | None:
    """Estimate the forward vanishing point and the camera pitch from one road image, or None when no lines
    converge cleanly (a featureless road)."""
    lines = _road_lines(image_bgr)
    vp = vanishing_point(lines)
    if vp is None:
        return None
    # a VP wildly outside the frame is an unreliable intersection; reject it
    if not (-image_bgr.shape[1] < vp[0] < 2 * image_bgr.shape[1] and 0 < vp[1] < image_bgr.shape[0]):
        return None
    return {"vp": [round(vp[0], 1), round(vp[1], 1)], "n_lines": len(lines),
            "pitch_deg": round(math.degrees(pitch_from_vp(vp[1], fy, cy)), 3)}


async def estimate_session_calibration(session_id, max_frames: int = 12) -> dict:
    """Estimate calibration for each camera in a session from a sample of its frames, and store it as
    source=estimated. Focal comes from EXIF when present (else the nominal lens); pitch is the median of the
    per-frame vanishing-point estimates. Cameras whose frames yield no usable lines are left on nominal."""
    import cv2
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame
    from db.session import get_sessionmaker
    from services.calibration.resolve import nominal_calibration
    from services.calibration.store import upsert_calibration

    async with get_sessionmaker()() as db:
        cams = (await db.execute(
            select(Frame.cam_id).where(Frame.session_id == session_id).distinct())).scalars().all()
        store = get_object_store()
        out: dict = {}
        for cam_id in cams:
            frames = (await db.execute(
                select(Frame.img_uri, Frame.width, Frame.height)
                .where(Frame.session_id == session_id, Frame.cam_id == cam_id).limit(max_frames))).all()
            if not frames:
                continue
            w, h = frames[0][1], frames[0][2]
            nom = nominal_calibration(cam_id, w, h)
            pitches = []
            for img_uri, fw, fh in frames:
                try:
                    arr = np.frombuffer(store.get_bytes(img_uri), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    est = estimate_frame(img, fw / 2.0, fh / 2.0, nom.fy)
                    if est is not None:
                        pitches.append(est["pitch_deg"])
                except Exception as exc:  # noqa: BLE001  a single unreadable frame must not abort the camera
                    log.warning("estimate.frame_failed", uri=img_uri, error=str(exc))
            if not pitches:
                out[cam_id] = {"stored": False, "reason": "no usable road lines"}
                continue
            pitch_deg = float(np.median(pitches))
            fields = {
                "model": nom.model, "fx": nom.fx, "fy": nom.fy, "cx": nom.cx, "cy": nom.cy,
                "dist": list(nom.dist), "ref_width": w,
                "rpy_deg": [0.0, pitch_deg, nom.rpy_deg[2]], "xyz_m": list(nom.xyz_m),
            }
            res = await upsert_calibration(session_id, cam_id, fields, "estimated")
            out[cam_id] = {**res, "pitch_deg": round(pitch_deg, 3), "frames_used": len(pitches)}
    log.info("calibration.estimated", session=str(session_id), cameras=len(out))
    return {"session_id": str(session_id), "source": "estimated", "cameras": out}
