"""Vegetation extraction: cluster trees and bushes as landmarks, excluded from the drivable surface. Uses the
Phase 2 vegetation semantic label when present, else falls back to tall, wide, non-planar non-ground clusters.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.lidar.extract.common import cluster_dbscan, cluster_stats, nonground_mask
from services.lidar.ingest.normalize import Cloud


def extract_vegetation(cloud: Cloud, semantic: np.ndarray | None, plane: list[float],
                       veg_class_ids: list[int] | None = None) -> list[dict]:
    """Tall, bushy (not thin, not flat) clusters are vegetation. Returns landmark positions, not drivable."""
    cfg = get_settings().lidar
    if semantic is not None and veg_class_ids:
        pts = cloud.xyz[np.isin(semantic, veg_class_ids)]
    else:
        pts = cloud.xyz[nonground_mask(cloud, plane)]
    if len(pts) < cfg.extract_cluster_min_points:
        return []
    labels = cluster_dbscan(pts, eps=0.7)
    veg = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = pts[labels == cl]
        st = cluster_stats(cxyz, plane)
        # vegetation is raised but not a thin pole and not a flat facade: wide in both footprint axes
        if st["height"] > 1.5 and st["footprint"] > 1.0 and st["footprint_min"] > 0.8:
            veg.append({"kind": "vegetation", "position": st["centroid"], "height": round(st["height"], 2),
                        "radius": round(st["footprint"] / 2.0, 2), "n_points": st["n"],
                        "drivable": False, "method": "cluster"})
    return veg
