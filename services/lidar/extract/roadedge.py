"""Road edge, curb, shoulder, and median extraction from the road segmentation boundary and the height step
at the edge. The left and right boundaries of the road surface become HD map road edges; a height step just
outside the boundary marks a curb.
"""

from __future__ import annotations

import numpy as np

from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud


def extract_road_edges(cloud: Cloud, semantic: np.ndarray | None, road_class_id: int,
                       plane: list[float], bin_m: float = 2.0) -> list[dict]:
    """The left and right lateral extent of the road surface, binned by forward distance, plus a curb flag
    where the points just outside the edge step up."""
    if semantic is None:
        return []
    road = cloud.xyz[semantic == road_class_id]
    if len(road) < 50:
        return []
    fwd, lat = road[:, 0], road[:, 1]
    bins = np.arange(fwd.min(), fwd.max() + bin_m, bin_m)
    left, right = [], []
    for i in range(len(bins) - 1):
        m = (fwd >= bins[i]) & (fwd < bins[i + 1])
        if m.sum() < 3:
            continue
        fx = float((bins[i] + bins[i + 1]) / 2.0)
        left.append([fx, float(lat[m].max())])
        right.append([fx, float(lat[m].min())])

    # curb: non-ground points just beyond the mean edge step up from the road plane
    nonroad = cloud.xyz[semantic != road_class_id] if semantic is not None else cloud.xyz
    above = height_above_plane(nonroad, plane)
    out = []
    for side, line in (("left", left), ("right", right)):
        if len(line) < 2:
            continue
        edge_lat = float(np.median([p[1] for p in line]))
        near = nonroad[np.abs(nonroad[:, 1] - edge_lat) < 1.0]
        curb = bool(len(near) and float(np.median(above[np.abs(nonroad[:, 1] - edge_lat) < 1.0])) > 0.1)
        out.append({"kind": "road_edge", "side": side, "line": line, "curb": curb,
                    "method": "road_boundary"})
    return out
