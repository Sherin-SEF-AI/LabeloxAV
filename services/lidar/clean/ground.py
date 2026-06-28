"""Ground segmentation and removal. Patchwork++ where the package is available, with a RANSAC plane-fit
fallback in open3d. The road plane is fit and split off so downstream annotation sees objects isolated from
the surface. Raw is immutable: this returns derived clouds, it never edits the input.

The plane is validated to be ground-like (near-horizontal normal in the ego frame, x forward, y left, z up),
so a building wall or a truck side is not mistaken for the road.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_ground")


def _ransac_ground(xyz: np.ndarray, dist_thresh: float, max_iter: int) -> tuple[np.ndarray, list[float]]:
    """Largest plane by RANSAC, accepted as ground only if its normal is near-vertical. Returns an inlier
    boolean mask over xyz and the plane coefficients [a, b, c, d] for ax+by+cz+d=0."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    model, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=max_iter)
    a, b, c, d = model
    normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(normal) or 1.0
    verticality = abs(normal[2]) / norm   # 1.0 = perfectly horizontal plane (vertical normal)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    if verticality >= 0.8:
        mask[np.asarray(inliers, dtype=int)] = True
    else:
        log.info("lidar.ground_not_horizontal", verticality=round(float(verticality), 3))
    return mask, [float(a), float(b), float(c), float(d)]


def _patchwork_ground(xyz: np.ndarray) -> tuple[np.ndarray, list[float]] | None:
    """Patchwork++ ground segmentation when pypatchworkpp is installed, else None to signal the fallback."""
    try:
        import pypatchworkpp
    except Exception:
        return None
    params = pypatchworkpp.Parameters()
    pw = pypatchworkpp.patchworkpp(params)
    pw.estimateGround(xyz.astype(np.float32))
    ground_idx = np.asarray(pw.getGroundIndices(), dtype=int)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[ground_idx] = True
    return mask, [0.0, 0.0, 1.0, 0.0]


def segment_ground(cloud: Cloud, method: str | None = None, dist_thresh: float | None = None,
                   max_iter: int | None = None) -> tuple[np.ndarray, list[float], str]:
    """Return a ground boolean mask over the cloud, the plane coefficients, and the method actually used."""
    cfg = get_settings().lidar
    method = method or cfg.ground_method
    dist_thresh = dist_thresh if dist_thresh is not None else cfg.ground_dist_thresh_m
    max_iter = max_iter if max_iter is not None else cfg.ground_max_iter
    if cloud.n < 3:
        return np.zeros(cloud.n, dtype=bool), [0.0, 0.0, 1.0, 0.0], "none"
    if method == "patchwork":
        pw = _patchwork_ground(cloud.xyz)
        if pw is not None:
            return pw[0], pw[1], "patchwork++"
        log.info("lidar.patchwork_unavailable_fallback_ransac")
    mask, plane = _ransac_ground(cloud.xyz, dist_thresh, max_iter)
    return mask, plane, "ransac"


def remove_ground(cloud: Cloud, method: str | None = None, dist_thresh: float | None = None,
                  max_iter: int | None = None) -> dict:
    """Split the cloud into ground and non-ground derived clouds. Returns both plus the plane and method."""
    mask, plane, used = segment_ground(cloud, method, dist_thresh, max_iter)
    ground = cloud.take(mask)
    nonground = cloud.take(~mask)
    log.info("lidar.ground_removed", method=used, total=cloud.n, ground=ground.n, kept=nonground.n)
    return {"ground": ground, "nonground": nonground, "plane": plane, "method": used,
            "ground_points": ground.n, "kept_points": nonground.n}
