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
    fids = onto.fallback_ids()
    if not fids:
        raise ValueError(f"ontology {onto.version} has no fallback class for the road surface")
    return fids[0]


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


# A 3D segmenter's class names (SemanticKITTI, nuScenes, Waymo) to this ontology, most specific first. A name
# with no counterpart stays unlabelled rather than being forced into a wrong class.
SEGMENTER_TO_ONTOLOGY: dict[str, tuple[str, ...]] = {
    "car": ("vehicle_fallback", "sedan"), "truck": ("truck",), "other-vehicle": ("vehicle_fallback",),
    "bus": ("bus",), "trailer": ("multi_axle_trailer", "truck"), "construction_vehicle": ("construction_veh",),
    "construction-vehicle": ("construction_veh",), "bicycle": ("cycle",), "motorcycle": ("motorcycle",),
    "person": ("pedestrian",), "pedestrian": ("pedestrian",), "bicyclist": ("cyclist", "rider"),
    "motorcyclist": ("rider",), "road": ("road",), "driveable_surface": ("road",), "parking": ("parking_area",),
    "sidewalk": ("sidewalk",), "other-ground": ("non_drivable_fallback",), "other_flat": ("non_drivable_fallback",),
    "terrain": ("green_belt", "non_drivable_fallback"), "building": ("buildings",), "manmade": ("buildings",),
    "fence": ("fence",), "barrier": ("barrier",), "vegetation": ("vegetation",), "trunk": ("tree",),
    "pole": ("pole",), "traffic-sign": ("traffic_sign",), "traffic_cone": ("cone",),
}


def segmenter_class_map(names: list[str], onto: Ontology | None = None) -> np.ndarray:
    """Index i of a segmenter's class list -> ontology id (or BACKGROUND)."""
    onto = onto or get_ontology()
    out = np.full(len(names), BACKGROUND, dtype=np.int32)
    for i, name in enumerate(names):
        for cand in SEGMENTER_TO_ONTOLOGY.get(name.lower(), ()):
            try:
                out[i] = onto.by_name(cand).id
                break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    return out


def segment_ptv3(cloud: Cloud, ckpt: str | None = None) -> dict:
    """PTv3 semantic segmentation on the local GPU, when the model is installed (segment3d/ptv3_infer.py,
    weights under models/ptv3/). Raises SegmentationUnavailable with the reason when it is not, so the caller
    falls back to the projected segmentation and says why."""
    try:
        from services.lidar.segment3d import ptv3_infer
    except Exception as exc:
        raise SegmentationUnavailable(
            "PTv3 is not installed on this machine (services/lidar/segment3d/ptv3_infer.py is absent or does "
            f"not import: {exc})") from exc
    ok, reason = ptv3_infer.available()
    if not ok:
        raise SegmentationUnavailable(f"PTv3 unavailable: {reason}")
    if cloud.n < 64:
        # serialisation and pooling need a real scan; a handful of points is not one
        raise SegmentationUnavailable(f"PTv3 needs a real scan; this cloud has {cloud.n} points")
    cfg = get_settings().lidar
    strength = None if cloud.intensity is None else np.asarray(cloud.intensity, dtype=np.float32)
    # The checkpoint was trained on nuScenes, whose intensity/255 runs far darker than KITTI reflectance
    # (medians ~0.05 vs ~0.27); a bright cloud is scaled down to the training range (ptv3_infer docstring).
    scale = 0.3 if strength is not None and cloud.n and float(np.median(strength)) > 0.15 else 1.0
    # Absolute z is a feature: a cloud in a ground-origin ego frame (pseudo-LiDAR) is moved to sensor height.
    z_shift = 0.0 if cloud.frame in ("kitti_velo", "nuscenes_lidar", "sensor", "velodyne") else -1.8
    res = ptv3_infer.infer(np.asarray(cloud.xyz, dtype=np.float32), strength, z_shift=z_shift,
                           strength_scale=scale)
    names = list(res["class_names"])
    lut = segmenter_class_map(names)
    raw = np.asarray(res["labels"], dtype=np.int64)
    valid = (raw >= 0) & (raw < len(lut))
    semantic = np.full(cloud.n, BACKGROUND, dtype=np.int32)
    semantic[valid] = lut[raw[valid]]
    conf = np.asarray(res["conf"], dtype=np.float32)
    low = float((conf < cfg.seg_low_conf).mean()) if cloud.n else 0.0
    present = sorted({int(x) for x in np.unique(semantic) if x != BACKGROUND})
    return {"semantic": semantic, "instance": np.full(cloud.n, BACKGROUND, dtype=np.int32), "conf": conf,
            "low_conf_frac": low, "classes_present": present, "n_instances": 0,
            "raw_labels": raw.astype(np.int32), "class_names": names,
            "checkpoint": str(res.get("checkpoint") or ckpt or cfg.segmenter_ckpt)}
