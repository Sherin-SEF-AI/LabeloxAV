"""Per-point semantic and instance segmentation.

Two paths, by config. The native path is PTv3 via Pointcept (real LiDAR, dense), run on the burst node; it is
not installed locally and parks on the lidar_perception seam. The runnable path for the camera fleet projects
the 3D understanding onto the points: a point inside a gated 3D cuboid takes that object's ontology class and
instance, a point on the ground plane is road, and everything else is left unlabeled and flagged low
confidence. This reuses the lifted boxes and the Phase 1 ground plane rather than trusting a real-LiDAR
segmenter on pseudo-LiDAR density, exactly as the milestone warns.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import Ontology, get_ontology
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_semantic3d")

BACKGROUND = -1          # unlabeled / no 3D support, flagged for review


class SegmentationUnavailable(RuntimeError):
    """Raised when Pointcept/PTv3 is not present locally; the burst job queues it for the A100."""


def road_class_id(onto: Ontology | None = None) -> int:
    """The ontology class for the road surface, with a drivable fallback."""
    onto = onto or get_ontology()
    for name in ("road", "drivable_area", "drivable", "carriageway"):
        try:
            return onto.by_name(name).id
        except Exception:
            continue
    for fid in onto.fallback_ids():
        if "drivable" in onto.by_id(fid).name:
            return fid
    return onto.fallback_ids()[0]


def points_in_cuboid(points: np.ndarray, cuboid: dict) -> np.ndarray:
    """Boolean mask of points inside an oriented cuboid (centre, [L, W, H], yaw about ego up)."""
    c = np.asarray(cuboid["center"], dtype=np.float32)
    length, width, height = cuboid["dims"]
    yaw = float(cuboid["yaw"])
    rel = points - c
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    lx = rel[:, 0] * cyaw + rel[:, 1] * syaw       # rotate into the box-local frame
    ly = -rel[:, 0] * syaw + rel[:, 1] * cyaw
    return (np.abs(lx) <= length / 2) & (np.abs(ly) <= width / 2) & (np.abs(rel[:, 2]) <= height / 2)


def segment_projected(cloud: Cloud, cuboids: list[dict], ground_plane: list[float] | None,
                      onto: Ontology | None = None) -> dict:
    """Label every point from the 3D cuboids plus the ground. Returns per-point semantic class, instance id,
    and confidence, with the low-confidence fraction flagged for review."""
    onto = onto or get_ontology()
    cfg = get_settings().lidar
    n = cloud.n
    semantic = np.full(n, BACKGROUND, dtype=np.int32)
    instance = np.full(n, BACKGROUND, dtype=np.int32)
    conf = np.full(n, 0.3, dtype=np.float32)        # background default is low confidence

    if ground_plane is not None:
        a, b, c, d = ground_plane
        if abs(c) > 1e-6:
            above = cloud.xyz[:, 2] - (-(a * cloud.xyz[:, 0] + b * cloud.xyz[:, 1] + d) / c)
            gmask = np.abs(above) < 0.2
            semantic[gmask] = road_class_id(onto)
            conf[gmask] = 0.8

    # larger boxes first so a smaller object inside (e.g. a rider on a bike) wins the overlap
    order = sorted(range(len(cuboids)), key=lambda i: -float(np.prod(cuboids[i]["dims"])))
    for inst_idx in order:
        cub = cuboids[inst_idx]
        m = points_in_cuboid(cloud.xyz, cub)
        semantic[m] = int(cub["class_id"])
        instance[m] = inst_idx
        conf[m] = 1.0

    low_conf_frac = float((conf < cfg.seg_low_conf).mean()) if n else 0.0
    classes = sorted({int(x) for x in np.unique(semantic) if x != BACKGROUND})
    return {"semantic": semantic, "instance": instance, "conf": conf, "low_conf_frac": low_conf_frac,
            "classes_present": classes, "n_instances": int(len(cuboids))}


def segment_ptv3(cloud: Cloud, ckpt: str | None = None) -> dict:
    """PTv3 semantic segmentation via Pointcept. Requires the framework on the burst node; raises the seam
    locally."""
    cfg = get_settings().lidar
    ckpt = ckpt or cfg.segmenter_ckpt
    try:
        import pointcept  # noqa: F401
    except Exception as exc:
        raise SegmentationUnavailable(
            f"PTv3 segmentation ({ckpt}) needs Pointcept on the A100 burst node. Run via the lidar_perception "
            "job; the local worker will not execute it.") from exc
    raise SegmentationUnavailable("Pointcept present but the inference wrapper runs on the burst node only.")
