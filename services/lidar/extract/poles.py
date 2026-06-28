"""Pole extraction: cluster vertical thin structures from the non-ground points and classify by height into
street light, traffic signal, electric, or camera pole. These become HD map landmarks.
"""

from __future__ import annotations

from core.config import get_settings
from services.lidar.extract.common import cluster_dbscan, cluster_stats, nonground_mask
from services.lidar.ingest.normalize import Cloud


def _classify_pole(height: float) -> str:
    if height >= 8.0:
        return "electric"
    if height >= 5.0:
        return "street_light"
    if height >= 3.0:
        return "traffic_signal"
    return "camera"


def extract_poles(cloud: Cloud, plane: list[float]) -> list[dict]:
    """Tall, thin, vertical non-ground clusters are poles. Returns one element per pole."""
    cfg = get_settings().lidar
    ng = nonground_mask(cloud, plane)
    pts = cloud.xyz[ng]
    if len(pts) < cfg.extract_cluster_min_points:
        return []
    labels = cluster_dbscan(pts)
    poles = []
    for cl in sorted(set(labels.tolist()) - {-1}):
        cxyz = pts[labels == cl]
        st = cluster_stats(cxyz, plane)
        aspect = st["height"] / max(st["footprint"], 0.1)
        if (st["height"] >= cfg.pole_min_height_m and st["footprint"] <= cfg.pole_max_footprint_m
                and aspect >= 3.0):
            poles.append({"kind": "pole", "pole_type": _classify_pole(st["height"]),
                          "position": st["centroid"], "base_z": st["base_z"],
                          "height": round(st["height"], 2), "n_points": st["n"], "method": "dbscan_vertical"})
    return poles
