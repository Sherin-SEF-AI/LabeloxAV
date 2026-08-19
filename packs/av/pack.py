"""The AV domain pack: LabeloxAV's original behaviour, authored against the DomainPack contract.

Every surface here is built from the engine's own values (imported, not re-declared) so the pack is a faithful
mirror and byte-parity holds by construction. The golden regression harness (tests/test_golden_av_pack.py)
freezes these values; any future change to AV behaviour trips it. See docs/AV_ASSUMPTIONS.md for the file:line
origin of each value.

Import direction: this module imports the engine (allowed). The engine never imports this module statically;
it reaches it only through packs.registry (enforced by .importlinter).
"""

from __future__ import annotations

from core.config import get_settings
from packs.av.scene_model import MovingCameraSceneModelFactory
from packs.base import (
    AutoLabelProfile,
    EvalStrataSpec,
    ForgeTarget,
    GatePolicy,
    OntologySpec,
    Pack,
    PackManifest,
    PrivacyPlaneSpec,
    QualityProfile,
    RedactionTarget,
    RelationSpec,
    RootCauseSignature,
    SensorCheck,
    StratumDimension,
    make_safety_policy,
)

# Single source of truth: import the AV values that live in engine code today (all import-light modules).
from services.autolabel.ontology import (
    CUSTOM_ID_BASE,
    STUFF_L0,
    STUFF_NAMES,
    get_ontology,
)
from services.curation.slices import _SCENE_AXES
from services.forgyx.capabilities import _BACKENDS
from services.forgyx.cooptimize import TARGET_BUDGET_MS
from services.forgyx.run import _FORMAT
from services.forgyx.thermal import TARGET_THERMAL
from services.sanyx.rootcause import REMEDIATION

# The l1 superclasses that define a safety-critical class. This is the `{"vru","animal"}` literal the audit
# found copy-pasted across six+ files (govern/champion.py:26, recall/gate.py:12, training/safe_miou.py:13,
# autolabel/gate.py:20, dynamics/compute.py:33, flywheel/signals.py:26). Consolidated here; SEC-M3/M4 point
# those call sites at pack.safety_policy.
AV_SAFETY_L1 = frozenset({"vru", "animal"})

# The safety-critical classes, by NAME. Fixes latent bug #2: services/verdyx/safety_recall.py:10 lists them
# as 0-based ids {0,1,2,3,8}, which disagree with the governed 1-based YAML ids. Consumed in SEC-M4.
AV_CRITICAL_CLASSES = frozenset({"pedestrian", "rider", "motorcycle", "cycle", "cattle"})

# The domain preamble hardcoded at services/autolabel/paths/path_c_qwen3vl.py:64 (no config seam today). The
# parity test asserts the live prompt still begins with this; SEC-M5 makes path_c read it from here.
AV_VLM_PROMPT_TEMPLATE = (
    "You are labeling an object cropped from an Indian road scene for an autonomous-driving "
    "dataset. Identify the object and read its attributes."
)

# The auto-label path values now live here (the pack owns them); the three paths read them at runtime from
# pack.autolabel_profile. Moved verbatim from services/autolabel/paths/* so AV behaviour is byte-identical.

# Cross-superclass road actors always offered to the VLM so it can fix gross mislabels across superclasses.
AV_CROSS_ANCHORS = (
    "autorickshaw", "e_auto", "e_rickshaw", "motorcycle", "scooter", "cycle",
    "pedestrian", "rider", "cyclist", "sedan", "suv", "hatchback", "pickup",
    "truck", "lcv", "bus", "tempo", "water_tanker", "cattle", "dog",
    "push_cart", "vendor_handcart", "street_vendor",
)

# COCO class name -> ontology class name; unmapped COCO detections fold to object_fallback (never dropped).
AV_COCO_TO_ONTOLOGY = {
    "person": "pedestrian",
    "bicycle": "cycle",
    "motorcycle": "motorcycle",
    "car": "sedan",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_signal",
    "stop sign": "traffic_sign",
    "fire hydrant": "pole",
    "bench": "object_fallback",
    "cat": "object_fallback",
    "dog": "dog",
    "horse": "cattle",
    "sheep": "goat",
    "cow": "cattle",
    "elephant": "object_fallback",
    "bird": "object_fallback",
}

# Open-vocab synonyms mapped back to one ontology class, so long-tail rural classes get proposed.
#
# A class with no entry here is prompted with its own name and the underscores swapped for spaces, which for
# signs meant the single phrase "traffic sign" covering 21 distinct Indian designs, from a red octagon to a
# green destination board. The phrases below name the designs the corpus actually contains.
#
# Signs and advertising are kept lexically apart on purpose. While `hoarding` was stuff the autolabeller
# dropped it and its contents were labelled `traffic_sign`; now that it is a thing, the two classes would
# compete for the same detections if their prompts both said "sign". So the sign phrases never say
# "billboard" or "advertisement", and the advertising phrases never say "sign". Tests hold both directions.
AV_OPENVOCAB_SYNONYMS = {
    "cattle": ("cattle", "cow", "buffalo", "bull", "ox"),
    "traffic_sign": (
        "traffic sign", "road sign", "a red and white triangular warning sign",
        "a circular speed limit sign", "a red octagonal stop sign",
        "a blue circular mandatory direction sign", "a green destination board with place names",
    ),
    "hoarding": (
        "advertising hoarding", "billboard", "a large commercial advertisement board",
        "a shop name board", "a fuel station price display",
    ),
    "traffic_signal": ("traffic signal", "traffic light", "a signal head with red amber green lamps"),
}


def _forge_targets() -> tuple[ForgeTarget, ...]:
    """Reassemble the AV silicon registries (services/forgyx/*) into ForgeTargets. The four deployment
    targets are those with a full thermal+latency profile; onnx/onnxruntime are intermediate backends."""
    out: list[ForgeTarget] = []
    for name in sorted(TARGET_BUDGET_MS):
        thermal = TARGET_THERMAL[name]
        out.append(ForgeTarget(
            name=name,
            backend=name,
            backend_modules=tuple(_BACKENDS.get(name, ())),
            throttle_temp_c=float(thermal["throttle_temp_c"]),
            power_ceiling_w=float(thermal["power_ceiling_w"]),
            latency_budget_ms=float(TARGET_BUDGET_MS[name]),
            export_format=_FORMAT.get(name, name),
        ))
    return tuple(out)


def _quality_profile() -> QualityProfile:
    """SANYX profile: the reused image checks + the AV inertial checks (limits from checks.py:167) + the AV
    fault table (services/sanyx/rootcause.py REMEDIATION)."""
    return QualityProfile(
        image_checks=("dropped_frames", "exposure", "lens_contamination"),
        sensor_checks=(
            SensorCheck(name="time_sync"),
            SensorCheck(name="gps"),
            # Xsens MTi-3 full-scale limits (services/sanyx/checks.py:167 defaults).
            SensorCheck(name="imu", params={"accel_limit": 156.9, "gyro_limit": 34.9}),
        ),
        rootcause_signatures=tuple(
            RootCauseSignature(name=k, remedy=v) for k, v in sorted(REMEDIATION.items())
        ),
    )


def _relations() -> RelationSpec:
    """The AV relationship vocabulary, unified across the two that were live and disjoint.

    The editor validated {rider_of, towed_by, part_of, member_of, occludes}; the scene-graph proposer wrote
    {occluded_by, following, crossing_in_front_of, parked_near} straight past that validation. `kinds` is
    the union, so a writer on either path is checkable against one list.

    `occludes` and `occluded_by` are the same fact in opposite directions and both were being stored, which
    means a query for either silently missed half the corpus. `occludes` is kept as canonical because it is
    the one the editor offers and the one a person draws.

    `overlap_pairs` is keyed on l1 superclasses rather than leaf classes, because the relation is a property
    of the kind of thing: every VRU heavily overlapping a two-wheeler is a rider on it, whatever the leaf
    class of either. This is what lets relationship-aware NMS keep a rider and their motorcycle as two
    objects instead of merging them, and say which relation justified it.
    """
    return RelationSpec(
        kinds=frozenset({"rider_of", "towed_by", "part_of", "member_of", "occludes",
                         "occluded_by", "following", "crossing_in_front_of", "parked_near"}),
        overlap_pairs={
            # A person on a two-wheeler or three-wheeler. The India case the editor was built around.
            ("vru", "two_wheeler"): "rider_of",
            ("vru", "three_wheeler"): "rider_of",
            # An animal drawing a cart, and a person pushing one: both are a VRU or animal in contact with
            # a vehicle that is not carrying them.
            ("animal", "cart"): "towed_by",
            ("vru", "cart"): "towed_by",
            # A person inside a four-wheeler, which overlaps heavily and is emphatically not one object.
            ("vru", "four_wheeler"): "part_of",
            ("vru", "heavy_vehicle"): "part_of",
        },
        inverse={"occluded_by": "occludes"},
    )


def _build() -> Pack:
    settings = get_settings()
    onto = get_ontology("av")

    # Fail loud if a critical class name is not in the governed ontology (guards against the id/name drift
    # that produced latent bug #2 in the first place).
    for name in AV_CRITICAL_CLASSES:
        if not onto.has_name(name):
            raise RuntimeError(f"AV critical class '{name}' is not in ontology {onto.version}")

    manifest = PackManifest(
        id="av",
        version="0.1.0",
        display_name="Autonomous Driving (India)",
        capabilities=frozenset({
            "moving_camera", "ego_motion", "dynamics", "mono_depth", "inertial_qa", "privacy_redaction",
        }),
        default_scene_model="moving_camera",
    )

    ontology = OntologySpec(
        yaml_path=str(settings.ontology_abspath()),
        stuff_names=STUFF_NAMES,
        stuff_l0=STUFF_L0,
        supported_core=frozenset(settings.ontology.supported_core),
        custom_id_base=CUSTOM_ID_BASE,
    )

    autolabel = AutoLabelProfile(
        vlm_prompt_template=AV_VLM_PROMPT_TEMPLATE,
        cross_anchors=AV_CROSS_ANCHORS,
        coco_map=dict(AV_COCO_TO_ONTOLOGY),
        openvocab_synonyms=dict(AV_OPENVOCAB_SYNONYMS),
        gate_policy=GatePolicy(safety_l1=AV_SAFETY_L1, rare_needs_review=True),
        disable_ego_hood_mask=False,
    )

    eval_strata = EvalStrataSpec(
        dimensions=tuple(StratumDimension(name=axis) for axis in _SCENE_AXES),
        protected_slices=("pedestrian_night", "autorickshaw_glare"),
        critical_class_names=AV_CRITICAL_CLASSES,
    )

    privacy = PrivacyPlaneSpec(
        redaction_targets=(
            RedactionTarget(name="face", detector="face"),
            RedactionTarget(name="plate", detector="plate"),
        ),
        legal_regime="DPDPA",
    )

    return Pack(
        manifest=manifest,
        ontology=ontology,
        safety_policy=make_safety_policy(onto, AV_SAFETY_L1, AV_CRITICAL_CLASSES),
        autolabel_profile=autolabel,
        eval_strata=eval_strata,
        quality_profile=_quality_profile(),
        forge_targets=_forge_targets(),
        privacy=privacy,
        relations=_relations(),
        scene_model=MovingCameraSceneModelFactory(),
        # The MCAP/CAN ingestion already lives in services/ingest (it fills vehicle_id + ego_speed); the AV
        # pack does not re-wrap it as an adapter yet. Sec ships the first IngestionAdapter (packs/sec).
        ingestion_adapters=(),
    )


PACK = _build()
