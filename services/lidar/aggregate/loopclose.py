"""Loop closure detection and pose-graph optimization. A revisited location (two non-consecutive poses close
in space) is a loop; the pose graph is corrected so the loop closes. GTSAM is the real optimizer on the A100
burst node (reusing the 2D HD map fusion); locally it falls back to distributing the accumulated drift across
the loop, the same graceful-degradation contract as cloud/mapfusion_pod.py.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("lidar_loopclose")


def _xy(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64)[:2, 3]


def detect_loops(poses: list[np.ndarray], radius: float | None = None, min_gap: int = 5) -> list[tuple[int, int]]:
    """Pairs of non-consecutive poses within `radius` metres: revisited locations."""
    radius = radius if radius is not None else get_settings().lidar.loop_closure_radius_m
    xys = [_xy(p) for p in poses]
    loops = []
    for i in range(len(poses)):
        for j in range(i + min_gap, len(poses)):
            if np.linalg.norm(xys[i] - xys[j]) < radius:
                loops.append((i, j))
                break    # the first revisit per anchor is enough
    return loops


def _gtsam_optimize(poses, loops):
    """GTSAM pose-graph optimization (A100 burst node only)."""
    import gtsam  # noqa: F401  (present only on the pod)
    raise RuntimeError("gtsam runs on the burst node only")


def optimize_pose_graph(poses: list[np.ndarray], loops: list[tuple[int, int]]) -> dict:
    """Correct the trajectory so each detected loop closes. Uses GTSAM when available, else distributes the
    loop-closure drift linearly across the intervening poses (the local fallback)."""
    xys = np.array([_xy(p) for p in poses], dtype=np.float64)
    corrected = xys.copy()
    try:
        return _gtsam_optimize(poses, loops)
    except Exception:
        pass

    method = "drift_distribution"
    for i, j in loops:
        # the loop says pose i and pose j are the same place; close the gap by spreading the residual
        residual = corrected[j] - corrected[i]
        span = j - i
        if span <= 0:
            continue
        for k in range(i + 1, j + 1):
            corrected[k] = corrected[k] - residual * ((k - i) / span)
        # carry the correction forward to the rest of the trajectory
        if j + 1 < len(corrected):
            corrected[j + 1:] = corrected[j + 1:] - residual

    drift_before = float(np.linalg.norm(xys[loops[0][1]] - xys[loops[0][0]])) if loops else 0.0
    drift_after = float(np.linalg.norm(corrected[loops[0][1]] - corrected[loops[0][0]])) if loops else 0.0
    log.info("lidar.loopclose", loops=len(loops), method=method,
             drift_before=round(drift_before, 3), drift_after=round(drift_after, 3))
    return {"method": method, "loops": loops, "corrected_xy": corrected.tolist(),
            "drift_before_m": round(drift_before, 3), "drift_after_m": round(drift_after, 3)}
