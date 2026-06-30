"""Point cloud quality checks: missing scan, sparsity, partial scan (a wide empty azimuth wedge), and, for
ring-bearing real LiDAR, dead channels. A failing cloud is flagged so its session is excluded from 3D
annotation, the same contract the 2D pipeline applies to a session that fails calibration.

Motion distortion (the sweep sheared by ego motion) needs per-point timestamps, which the single-shot
pseudo-LiDAR and the dataset scans here do not carry; it is reported as not evaluated rather than guessed.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_quality")


def _largest_empty_wedge_deg(xyz: np.ndarray, bins: int = 72) -> float:
    """The widest contiguous azimuth gap with no points, in degrees. A full 360 scan with a wide gap is a
    partial scan (a dropped sector); a forward-only cloud naturally has a large rear gap."""
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    hist, _ = np.histogram(az, bins=bins, range=(-np.pi, np.pi))
    occ = hist > 0
    if not occ.any():
        return 360.0
    # walk the circular occupancy for the longest run of empty bins
    doubled = np.concatenate([occ, occ])
    best = run = 0
    for v in doubled:
        run = 0 if v else run + 1
        best = max(best, run)
    best = min(best, bins)
    return round(best / bins * 360.0, 1)


def check_cloud_quality(cloud: Cloud, *, min_points: int | None = None, full_360: bool | None = None,
                        max_empty_wedge_deg: float | None = None, expected_rings: int | None = None) -> dict:
    """Run every applicable quality check and return per-check flags plus an overall status."""
    cfg = get_settings().lidar
    min_points = min_points if min_points is not None else cfg.quality_min_points
    max_wedge = max_empty_wedge_deg if max_empty_wedge_deg is not None else cfg.quality_max_empty_wedge_deg
    # only a sensor expected to see all around (real spinning LiDAR) can have a partial scan
    full_360 = full_360 if full_360 is not None else (cloud.source == "lidar")

    checks: dict[str, bool] = {}
    checks["missing_scan"] = cloud.n == 0
    checks["sparse"] = 0 < cloud.n < min_points
    empty_wedge = _largest_empty_wedge_deg(cloud.xyz) if cloud.n else 360.0
    checks["partial_scan"] = bool(full_360 and empty_wedge > max_wedge)

    dead_channels = 0
    if cloud.ring is not None and cloud.n:
        rings = np.unique(cloud.ring)
        span = int(rings.max() - rings.min() + 1)
        dead_channels = max(0, span - len(rings))
    checks["dead_channels"] = dead_channels > 0

    failed = checks["missing_scan"] or checks["sparse"] or checks["partial_scan"] or checks["dead_channels"]
    out = {"status": "fail" if failed else "pass", "checks": checks, "points": cloud.n,
           "largest_empty_wedge_deg": empty_wedge, "dead_channels": dead_channels,
           "motion_distortion": "not_evaluated (needs per-point timing)"}
    log.info("lidar.quality", status=out["status"], points=cloud.n, wedge=empty_wedge, dead=dead_channels)
    return out
