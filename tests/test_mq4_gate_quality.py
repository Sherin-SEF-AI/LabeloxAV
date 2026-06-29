"""M-Q.4: the quality reviewer demotes geometric/contextual nonsense, and the gate enforces per-class
calibrated thresholds plus a rare-class agreement+VLM rule. Together these stop confident-but-wrong
detections (sky boxes, impossible sizes, tyre-as-vehicle, duplicates, pedestrian-in-car, ungrounded rare
classes) from being auto-accepted. A 1280x720 frame is used so the horizon line is at y=324."""

from __future__ import annotations

from core.config import GateSettings, QualitySettings
from core.schemas import BBox, GateState, PathProposal, Provenance, UnifiedObject
from services.autolabel.gate import class_auto_accept, gate_object, vlm_confirmed
from services.autolabel.ontology import get_ontology
from services.autolabel.quality_reviewer import review_object_quality

ONTO = get_ontology()
W, H = 1280, 720


def _obj(name, bbox, conf=0.9, agreement=True, vlm=False, mbd=False):
    c = ONTO.by_name(name)
    props = ([PathProposal(path="path_c_qwen3vl", class_name=name, verdict="confirm", model_version="t")]
             if vlm else [])
    return UnifiedObject(class_id=c.id, class_name=name, bbox=BBox.from_list(bbox), conf=conf,
                         provenance=Provenance(agreement=agreement, mask_box_disagree=mbd, proposals=props))


def _flags(obj, others=()):
    return review_object_quality(obj, list(others), ONTO, W, H, QualitySettings()).reasons


# ---- quality reviewer rules --------------------------------------------------------------------------

def test_above_horizon_flags_sky_box():
    assert "above_horizon" in _flags(_obj("sedan", [600, 40, 720, 160]))


def test_impossible_size_tiny_heavy_and_huge_vru():
    assert "impossible_size" in _flags(_obj("truck", [600, 400, 612, 412]))   # a truck 12px wide
    assert "impossible_size" in _flags(_obj("pedestrian", [0, 0, 1200, 700]))  # a pedestrian filling the frame


def test_heavy_smaller_than_vru_is_flagged():
    truck = _obj("truck", [600, 400, 665, 465])
    ped = _obj("pedestrian", [100, 300, 360, 700])
    assert "smaller_than_vru" in _flags(truck, [ped])


def test_small_vehicle_inside_larger_is_part():
    wheel = _obj("sedan", [620, 420, 665, 465])
    truck = _obj("truck", [600, 400, 900, 700])
    assert "part_of_vehicle" in _flags(wheel, [truck])


def test_duplicate_box_demotes_the_lower_confidence_one():
    a = _obj("sedan", [600, 400, 700, 500], conf=0.90)
    b = _obj("sedan", [602, 402, 702, 502], conf=0.95)
    assert "duplicate_box" in _flags(a, [b])       # a overlaps a higher-conf box
    assert "duplicate_box" not in _flags(b, [a])   # b is the keeper


def test_pedestrian_inside_car_is_flagged():
    ped = _obj("pedestrian", [620, 440, 680, 580])
    car = _obj("sedan", [600, 400, 800, 620])
    assert "vru_inside_vehicle" in _flags(ped, [car])


def test_clean_object_passes():
    assert _flags(_obj("sedan", [500, 400, 650, 560])) == []


def test_optional_context_off_road_and_scene():
    qs = QualitySettings()
    car = _obj("sedan", [500, 400, 650, 560])
    assert "off_road" in review_object_quality(car, [], ONTO, W, H, qs, on_drivable=False).reasons
    assert "implausible_in_scene" in review_object_quality(car, [], ONTO, W, H, qs, scene="indoor").reasons


# ---- gate: per-class thresholds + rare agreement+VLM + quality verdict -------------------------------

def test_per_class_threshold_values():
    cfg = GateSettings()
    assert class_auto_accept(ONTO.by_name("pedestrian").id, ONTO, cfg) == cfg.safety_auto_accept  # 0.99
    assert class_auto_accept(ONTO.by_name("sedan").id, ONTO, cfg) == cfg.auto_accept              # 0.95


def test_benign_auto_accepts_at_default_but_safety_needs_higher():
    cfg = GateSettings()
    assert gate_object(_obj("sedan", [10, 400, 60, 460], conf=0.96), ONTO, cfg) == GateState.auto_accept
    # a pedestrian at 0.96 is below the 0.99 safety floor -> review, not auto-accept
    assert gate_object(_obj("pedestrian", [10, 400, 40, 500], conf=0.96), ONTO, cfg) == GateState.review
    assert gate_object(_obj("pedestrian", [10, 400, 40, 500], conf=0.995), ONTO, cfg) == GateState.auto_accept


def test_rare_class_needs_agreement_and_vlm():
    cfg = GateSettings()
    # autorickshaw is India-specific (rare): agreement alone is not enough
    assert gate_object(_obj("autorickshaw", [10, 400, 90, 500], conf=0.99, vlm=False), ONTO, cfg) == GateState.review
    assert gate_object(_obj("autorickshaw", [10, 400, 90, 500], conf=0.99, vlm=True), ONTO, cfg) == GateState.auto_accept
    assert vlm_confirmed(_obj("autorickshaw", [0, 0, 1, 1], vlm=True).provenance)


def test_quality_verdict_demotes_even_a_confident_box():
    cfg = GateSettings()
    obj = _obj("sedan", [10, 400, 60, 460], conf=0.99)
    assert gate_object(obj, ONTO, cfg, quality_ok=True) == GateState.auto_accept
    assert gate_object(obj, ONTO, cfg, quality_ok=False) == GateState.review
