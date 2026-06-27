"""Intrinsics validation (M3.0). The FOV check is the key defence against the narrow-vs-wide lens-mixing
issue: a camera configured as one lens type but carrying the other's intrinsics has an implied field of
view far from the configured one and fails. Reprojection error comes from a ChArUco capture when present
(the wide STURDeCAM31 lenses use the cv2.fisheye model, narrow use pinhole).
"""

from __future__ import annotations

import math

import numpy as np

from core.config import LensIntrinsics, get_settings
from core.logging import get_logger

log = get_logger("calib_intrinsics")


def K_from(lens: LensIntrinsics) -> np.ndarray:
    return np.array([[lens.fx, 0.0, lens.cx], [0.0, lens.fy, lens.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def implied_hfov_deg(lens: LensIntrinsics, ref_width: int) -> float:
    """Horizontal FOV implied by the intrinsics, model-aware: pinhole is 2 atan(W/2fx); fisheye is the
    equidistant projection W/fx (radians). Intrinsics are defined at the reference width."""
    if lens.model == "fisheye":
        return math.degrees(ref_width / max(lens.fx, 1e-6))
    return math.degrees(2.0 * math.atan(ref_width / (2.0 * max(lens.fx, 1e-6))))


def fov_check(actual: LensIntrinsics, configured_lens: str) -> dict:
    """Compare the FOV implied by the actual intrinsics against the configured lens. A mismatch (a narrow
    camera carrying wide intrinsics, or vice versa) exceeds the tolerance and fails. This is the lens-mix
    catch."""
    cfg = get_settings()
    ref = cfg.rig.ref_width
    expected = cfg.rig.lenses[configured_lens].fov_deg if configured_lens in cfg.rig.lenses else None
    implied = implied_hfov_deg(actual, ref)
    diff = abs(implied - expected) if expected is not None else None
    ok = diff is not None and diff <= cfg.spatial.fov_tolerance_deg
    return {"configured_lens": configured_lens, "expected_fov_deg": round(expected, 1) if expected else None,
            "implied_fov_deg": round(implied, 1), "diff_deg": round(diff, 1) if diff is not None else None,
            "tolerance_deg": cfg.spatial.fov_tolerance_deg, "ok": bool(ok)}


def reprojection_error_charuco(image_bgr: np.ndarray, lens: LensIntrinsics,
                               squares_x: int = 8, squares_y: int = 6,
                               square_len: float = 0.04, marker_len: float = 0.03) -> float | None:
    """Mean reprojection error (px) over a detected ChArUco board, or None if no board is found."""
    import cv2

    K = K_from(lens)
    dist = np.asarray(lens.dist, dtype=np.float64)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, adict)
    try:
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    except Exception:  # noqa: BLE001
        return None
    if ch_corners is None or len(ch_corners) < 6:
        return None
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist if dist.size else None)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist if dist.size else None)
    return float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())


def validate_intrinsics(actual: LensIntrinsics, configured_lens: str,
                        charuco_image: np.ndarray | None = None) -> dict:
    fov = fov_check(actual, configured_lens)
    reproj = reprojection_error_charuco(charuco_image, actual) if charuco_image is not None else None
    return {"model": actual.model, "fov_check": fov, "reproj_error_px": reproj}
