"""M-Q.4 quality reviewer: geometric and contextual checks that demote nonsense before it reaches the
auto-accept queue. Each rule catches a real error observed live (a sky/wall box, an impossible size, a
tyre labeled a vehicle, a duplicate box, a pedestrian inside a car). The checks are deterministic and
model-free, so they cost nothing and run on every object; a flagged object is routed to human review and
never silently auto-accepted, and the reasons are recorded for the correction-and-retrain loop.

Off-road context and scene plausibility need the drivable segmentation / scene class; they are applied when
that context is provided and skipped otherwise, so the reviewer degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import QualitySettings
from core.schemas import BBox, UnifiedObject
from services.autolabel.ontology import Ontology

# superclasses that live on the road plane (so a box entirely in the sky is wrong)
_GROUND = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy", "vru", "animal"}
_VEHICLE = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy"}
_BIG_VEHICLE = {"four_wheeler", "heavy"}
_VRU = {"vru"}
# classes that legitimately sit high in the frame (overhead), exempt from the horizon rule
_OVERHEAD = {"traffic_signal", "traffic_sign", "pole", "street_light", "sign_board", "hoarding"}


@dataclass
class QualityVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _contained_frac(inner: BBox, outer: BBox) -> float:
    """Fraction of inner's area that lies inside outer."""
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = inner.area
    return inter / a if a > 0 else 0.0


def review_object_quality(
    obj: UnifiedObject, others: list[UnifiedObject], onto: Ontology,
    frame_w: int, frame_h: int, cfg: QualitySettings,
    *, on_drivable: bool | None = None, scene: str | None = None,
) -> QualityVerdict:
    """Run the geometric/contextual checks on one object. others is the rest of the frame's fused objects.
    on_drivable / scene are optional context (drivable-area test, scene plausibility) applied when given."""
    if not cfg.enabled:
        return QualityVerdict(ok=True)

    reasons: list[str] = []
    name = onto.by_id(obj.class_id)
    l1 = name.l1
    bb = obj.bbox
    frame_area = max(1.0, float(frame_w) * float(frame_h))
    area_frac = bb.area / frame_area

    # 1. Above the horizon: a ground object whose box sits entirely in the sky region cannot be a road
    #    object (catches sky and building-wall detections). Overhead infra (signs, poles) is exempt.
    if l1 in _GROUND and name.name not in _OVERHEAD and bb.y2 < cfg.horizon_frac * frame_h:
        reasons.append("above_horizon")

    # 2. Impossible size: outside the plausible per-superclass area band, or a heavy vehicle smaller than a
    #    VRU in the same frame (a bus smaller than a bicycle).
    bounds = cfg.size_bounds.get(l1)
    if bounds and not (bounds[0] <= area_frac <= bounds[1]):
        reasons.append("impossible_size")
    if l1 == "heavy":
        for o in others:
            if onto.by_id(o.class_id).l1 in _VRU and o.bbox.area > bb.area * 1.2:
                reasons.append("smaller_than_vru")
                break

    # 4. Part versus whole: a small vehicle box mostly inside a much larger vehicle box is a part of it (a
    #    wheel/tyre read as a vehicle), not its own vehicle.
    if l1 in _VEHICLE:
        for o in others:
            if o is obj:
                continue
            if (onto.by_id(o.class_id).l1 in _VEHICLE and o.bbox.area > bb.area * 2.0
                    and _contained_frac(bb, o.bbox) > cfg.part_contain):
                reasons.append("part_of_vehicle")
                break

    # 5. Duplicate / heavy overlap on the same pixels: keep the higher-confidence box, demote this one.
    for o in others:
        if o is obj:
            continue
        if _iou(bb, o.bbox) > cfg.dup_iou and o.conf >= obj.conf:
            reasons.append("duplicate_box")
            break

    # 6. A VRU inside a four-wheeler / heavy box is implausible (a reflection or a misdetection on glass); a
    #    rider on a two-wheeler is legitimate, so only big vehicles trigger this.
    if l1 in _VRU:
        for o in others:
            if o is obj:
                continue
            if onto.by_id(o.class_id).l1 in _BIG_VEHICLE and _contained_frac(bb, o.bbox) > cfg.vru_contain:
                reasons.append("vru_inside_vehicle")
                break

    # 3. Off-road context (optional): a road object entirely off the drivable surface is suspect.
    if on_drivable is False and l1 in _VEHICLE:
        reasons.append("off_road")

    # 7. Scene plausibility (optional): a class that cannot appear in the classified scene.
    if scene and l1 in _VEHICLE and scene in ("indoor", "building_interior"):
        reasons.append("implausible_in_scene")

    return QualityVerdict(ok=len(reasons) == 0, reasons=reasons)
