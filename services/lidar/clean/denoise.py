"""Noise filtering: statistical and radius outlier removal (open3d), plus a dedicated rain-and-dust pass.
Rain and dust show up as sparse, low-intensity returns close to the sensor; the pass removes points that are
both poorly supported by neighbours and low intensity, so real low-reflectance surfaces are kept. Raw is
immutable: every pass returns a derived cloud.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_denoise")


def _to_o3d(xyz: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


def statistical_outlier(cloud: Cloud, nb_neighbors: int | None = None,
                        std_ratio: float | None = None) -> Cloud:
    """Drop points whose mean neighbour distance is an outlier versus the cloud, the classic LiDAR denoiser."""
    cfg = get_settings().lidar
    nb = nb_neighbors if nb_neighbors is not None else cfg.denoise_nb_neighbors
    ratio = std_ratio if std_ratio is not None else cfg.denoise_std_ratio
    if cloud.n <= nb:
        return cloud
    _, keep = _to_o3d(cloud.xyz).remove_statistical_outlier(nb_neighbors=nb, std_ratio=ratio)
    return cloud.take(np.asarray(keep, dtype=int))


def radius_outlier(cloud: Cloud, min_points: int | None = None, radius: float | None = None) -> Cloud:
    """Drop points with too few neighbours inside a radius, removing isolated speckle."""
    cfg = get_settings().lidar
    min_pts = min_points if min_points is not None else cfg.denoise_min_points
    rad = radius if radius is not None else cfg.denoise_radius_m
    if cloud.n <= min_pts:
        return cloud
    _, keep = _to_o3d(cloud.xyz).remove_radius_outlier(nb_points=min_pts, radius=rad)
    return cloud.take(np.asarray(keep, dtype=int))


def filter_rain_dust(cloud: Cloud, intensity_pct: float = 25.0, radius: float | None = None,
                     min_neighbors: int = 4) -> Cloud:
    """Remove rain and dust: points that are both low intensity and sparsely supported. A point survives if
    its intensity is above the low percentile OR it has enough close neighbours, so real dark surfaces and
    dense structure are preserved."""
    cfg = get_settings().lidar
    rad = radius if radius is not None else cfg.denoise_radius_m
    if cloud.n <= min_neighbors:
        return cloud
    import open3d as o3d

    pcd = _to_o3d(cloud.xyz)
    tree = o3d.geometry.KDTreeFlann(pcd)
    counts = np.empty(cloud.n, dtype=np.int32)
    for i in range(cloud.n):
        k, _, _ = tree.search_radius_vector_3d(pcd.points[i], rad)
        counts[i] = k - 1  # exclude the point itself
    thresh = np.percentile(cloud.intensity, intensity_pct) if cloud.intensity.size else 0.0
    bright = cloud.intensity > thresh
    supported = counts >= min_neighbors
    keep = bright | supported
    log.info("lidar.rain_dust_filtered", total=cloud.n, removed=int((~keep).sum()))
    return cloud.take(keep)


def denoise(cloud: Cloud, rain_dust: bool = True) -> Cloud:
    """The full noise pass: statistical then radius outlier removal, then optional rain and dust filtering."""
    out = statistical_outlier(cloud)
    out = radius_outlier(out)
    if rain_dust:
        out = filter_rain_dust(out)
    log.info("lidar.denoised", total=cloud.n, kept=out.n)
    return out
