"""Building extraction: fit near-vertical facade planes to large non-ground clusters; a tall vertical plane
is a building facade, and its base line is the HD map building boundary.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.lidar.extract.common import cluster_dbscan, cluster_stats, nonground_mask
from services.lidar.ingest.normalize import Cloud


def extract_buildings(cloud: Cloud, plane: list[float]) -> list[dict]:
    """Large, tall, near-vertical planar clusters are building facades. Returns the facade base line."""
    cfg = get_settings().lidar
    min_facade = cfg.building_min_facade_points
    ng = nonground_mask(cloud, plane, thresh=0.5)
    pts = cloud.xyz[ng]
    if len(pts) < min_facade:
        return []
    import open3d as o3d

    # cluster with the default density threshold, then filter by cluster SIZE (min_facade is a count of
    # points in a facade, not a DBSCAN core-density threshold)
    labels = cluster_dbscan(pts, eps=1.0)
    buildings = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = pts[labels == cl]
        if len(cxyz) < min_facade:
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cxyz.astype(np.float64))
        model, inliers = pc.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=100)
        a, b, c, _ = model
        norm = float(np.linalg.norm([a, b, c])) or 1.0
        verticality = 1.0 - abs(c) / norm          # 1.0 for a vertical plane (horizontal normal)
        st = cluster_stats(cxyz, plane)
        if verticality > 0.8 and st["height"] > 2.5 and len(inliers) > min_facade * 0.5:
            facade = cxyz[np.asarray(inliers, dtype=int)]
            line = [[float(facade[:, 0].min()), float(facade[:, 1].min())],
                    [float(facade[:, 0].max()), float(facade[:, 1].max())]]
            buildings.append({"kind": "building", "footprint": line, "height": round(st["height"], 2),
                              "n_points": st["n"], "verticality": round(verticality, 3),
                              "method": "facade_plane"})
    return buildings
