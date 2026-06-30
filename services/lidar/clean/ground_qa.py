"""Milestone F: ground-plane labeling review. The traversability pipeline already fits a ground plane and
derives the drivable surface automatically (services/lidar/traverse, DrivableMask). The missing piece in the
labeling loop is knowing which clouds the auto fit cannot be trusted on, so a human labels those by hand.
This classifies the fitted plane as ok / tilted / sparse / absent and flags the ones that need review. The
verticality of plane ax+by+cz+d=0 in the ego frame (z up) is |c| / norm; a true ground plane is near 1.0.
"""

from __future__ import annotations

import math

import numpy as np

from core.logging import get_logger

log = get_logger("ground_qa")


def ground_plane_status(plane: list[float], ground_frac: float, n_points: int, *, min_frac: float = 0.15,
                        min_points: int = 500, vert_thresh: float = 0.92) -> dict:
    """Classify a fitted ground plane. absent: no usable plane or no ground inliers. tilted: the normal is
    too far from vertical to be a road surface. sparse: too few points or too little ground to trust the fit.
    ok: otherwise. needs_review is true for anything but ok."""
    a, b, c, _ = plane
    norm = math.sqrt(a * a + b * b + c * c)
    verticality = abs(c) / norm if norm > 1e-9 else 0.0
    if norm < 1e-9 or ground_frac <= 0.0:
        status = "absent"
    elif verticality < vert_thresh:
        status = "tilted"
    elif n_points < min_points or ground_frac < min_frac:
        status = "sparse"
    else:
        status = "ok"
    return {"status": status, "verticality": round(verticality, 4), "ground_frac": round(ground_frac, 4),
            "n_points": int(n_points), "needs_review": status != "ok"}


async def flag_ground_for_review(cloud_id, dist_thresh: float = 0.2) -> dict:
    """Load the cloud and its fitted plane, measure the ground inlier fraction, and return the review flag."""
    from services.lidar.extract.common import load_for_extraction
    data = await load_for_extraction(cloud_id)
    if data is None:
        return {"error": "cloud not found"}
    cloud, plane = data["cloud"], data["plane"]
    a, b, c, d = plane
    norm = math.sqrt(a * a + b * b + c * c) or 1.0
    dist = np.abs(cloud.xyz @ np.array([a, b, c]) + d) / norm
    ground_frac = float(np.mean(dist < dist_thresh)) if cloud.n else 0.0
    res = ground_plane_status(plane, ground_frac, cloud.n)
    log.info("ground_qa.flag", cloud=str(cloud_id), status=res["status"], needs_review=res["needs_review"])
    return {"cloud_id": str(cloud_id), **res}
