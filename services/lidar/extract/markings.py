"""Road marking extraction: high-intensity returns on the road surface are paint. Cluster them and classify
by shape into stop lines (lateral), lane boundaries (longitudinal), and crosswalks.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from services.lidar.extract.common import cluster_dbscan, cluster_stats
from services.lidar.ingest.normalize import Cloud


def extract_markings(cloud: Cloud, semantic: np.ndarray | None, road_class_id: int,
                     plane: list[float]) -> list[dict]:
    """High-intensity road points clustered and classified by footprint shape."""
    cfg = get_settings().lidar
    if semantic is None:
        return []
    road = semantic == road_class_id
    if road.sum() < 50:
        return []
    inten = cloud.intensity[road]
    rxyz = cloud.xyz[road]
    thr = float(np.percentile(inten, cfg.marking_intensity_pct))
    bright = rxyz[inten > thr]
    if len(bright) < cfg.extract_cluster_min_points:
        return []
    labels = cluster_dbscan(bright, eps=0.6)
    markings = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = bright[labels == cl]
        st = cluster_stats(cxyz, plane)
        dx = st["bounds_max"][0] - st["bounds_min"][0]   # forward extent
        dy = st["bounds_max"][1] - st["bounds_min"][1]   # lateral extent
        if dy > 2.0 and dy > 2.0 * dx:
            mtype = "stop_line"
        elif dx > 3.0 and dx > 2.0 * dy:
            mtype = "lane_boundary"
        else:
            mtype = "crosswalk"
        markings.append({"kind": mtype, "position": st["centroid"],
                         "extent": [round(dx, 2), round(dy, 2)], "n_points": st["n"],
                         "method": "intensity_road"})
    return markings
