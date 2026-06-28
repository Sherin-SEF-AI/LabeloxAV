"""Road surface classification from intensity and geometry: asphalt, concrete, gravel, mud, sand, or water.
This matters for Indian unpaved and waterlogged roads. Water returns are specular (very low intensity);
unpaved surfaces are rough (high height residual from the ground plane); concrete is bright and smooth;
asphalt is mid intensity and smooth. The camera texture is a documented refinement.
"""

from __future__ import annotations

import numpy as np

from services.lidar.extract.common import height_above_plane
from services.lidar.ingest.normalize import Cloud


def classify_surface(cloud: Cloud, semantic: np.ndarray | None, road_class_id: int,
                     plane: list[float]) -> dict:
    """Classify the road surface from the road points' intensity statistics and roughness."""
    if semantic is not None:
        mask = semantic == road_class_id
    else:
        mask = np.abs(height_above_plane(cloud.xyz, plane)) < 0.15   # near-ground fallback
    rpts = cloud.xyz[mask]
    rinten = cloud.intensity[mask]
    if len(rpts) < 50:
        return {"surface": "unknown", "n_points": int(len(rpts)), "confidence": 0.0}

    mean_i = float(rinten.mean())
    roughness = float(np.std(height_above_plane(rpts, plane)))   # height scatter from the road plane
    if mean_i < 0.08:
        surface = "water"                                  # specular: almost no return
    elif roughness > 0.06:
        surface = "mud" if mean_i < 0.3 else "gravel"      # rough unpaved
    elif mean_i > 0.6:
        surface = "concrete"                               # bright and smooth
    else:
        surface = "asphalt"                                # mid intensity, smooth
    confidence = round(min(1.0, len(rpts) / 500.0), 2)
    return {"surface": surface, "mean_intensity": round(mean_i, 3), "roughness": round(roughness, 4),
            "n_points": int(len(rpts)), "confidence": confidence}
