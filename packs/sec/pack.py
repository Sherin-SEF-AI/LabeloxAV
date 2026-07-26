"""The LabeloxSec domain pack: India CCTV / security footage, authored against the DomainPack contract.

Every surface is real. The scene model (static camera), ingestion adapter, and quality profile are the M2
components; the ontology is the governed ontology/sec/labelox_sec_v0.yaml; the safety definition is
person-and-weapon (not the AV VRU/animal); forge targets are empty until the CCTV silicon profiles land in
SEC-M9. The Sec digest is frozen by the golden harness alongside AV's.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import yaml

from core.config import REPO_ROOT
from packs.base import (
    AutoLabelProfile,
    EvalStrataSpec,
    GatePolicy,
    OntologySpec,
    Pack,
    PackManifest,
    PrivacyPlaneSpec,
    RedactionTarget,
    StratumDimension,
    make_safety_policy,
    superclass_affinity_cost,
)
from packs.sec.ingest import StaticCameraIngestionAdapter
from packs.sec.quality import build_quality_profile
from packs.sec.scene_model import StaticCameraSceneModelFactory
from services.autolabel.ontology import CUSTOM_ID_BASE, load_ontology

_PACK_DIR = Path(__file__).resolve().parent
_META = yaml.safe_load((_PACK_DIR / "pack.yaml").read_text())
_ONTOLOGY_PATH = str(REPO_ROOT / _META["ontology"])

# The security-critical superclasses. A person or a weapon is never auto-dismissed. This is Sec's analogue of
# AV's `{"vru","animal"}`; it lives here, once, not copy-pasted across the engine.
SEC_SAFETY_L1 = frozenset({"person", "weapon"})

# The highest-priority security actors, by name (the surface SEC-M4 consumes for critical-recall gating).
SEC_CRITICAL_CLASSES = frozenset({"person", "weapon", "firearm", "knife", "abandoned_object"})

# Fixed scene infrastructure: stuff, never a tracked instance in the usual flow (surfaces are stuff by l0).
SEC_STUFF_NAMES = frozenset({
    "gate", "door", "barrier", "fence", "turnstile", "boom_barrier", "cctv_camera", "signage",
})

# The grounded actors the open-vocab / VLM paths are prompted with (mirrors AV's supported_core discipline).
SEC_SUPPORTED_CORE = frozenset({
    "person", "security_guard", "worker", "child", "delivery_person", "vendor",
    "dog", "cattle",
    "car", "motorcycle", "bicycle", "autorickshaw", "truck", "bus", "e_rickshaw",
    "weapon", "knife", "firearm", "lathi",
    "bag", "backpack", "suitcase", "abandoned_object",
})

SEC_VLM_PROMPT_TEMPLATE = (
    "You are labeling an object cropped from an Indian CCTV / security camera scene. Identify the object and "
    "read its attributes."
)

# Cross-superclass anchors the VLM can use to fix gross mislabels (a bag read as a person, a lathi as a stick).
SEC_CROSS_ANCHORS = (
    "person", "security_guard", "worker", "delivery_person", "child",
    "bag", "backpack", "suitcase", "abandoned_object",
    "weapon", "knife", "firearm", "lathi",
    "dog", "cattle", "car", "motorcycle", "autorickshaw", "bicycle", "truck", "bus",
)

# COCO class name -> Sec ontology class name. Unmapped COCO detections fold to object_fallback (never dropped).
SEC_COCO_TO_ONTOLOGY = {
    "person": "person",
    "backpack": "backpack",
    "handbag": "bag",
    "suitcase": "suitcase",
    "knife": "knife",
    "car": "car",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "bus": "bus",
    "truck": "truck",
    "dog": "dog",
    "cat": "cat",
}

# Open-vocab synonyms mapped back to one Sec class, so the long-tail security items actually get proposed.
SEC_OPENVOCAB_SYNONYMS = {
    "weapon": ("weapon", "gun", "pistol", "rifle", "blade"),
    "firearm": ("firearm", "gun", "pistol", "rifle", "handgun"),
    "abandoned_object": ("unattended bag", "abandoned baggage", "unattended object", "suspicious package"),
    "lathi": ("lathi", "baton", "stick", "cane"),
}


def _build() -> Pack:
    # Load the ontology directly (not get_ontology("sec"), which would re-enter the registry that is mid-build).
    onto = load_ontology(_ONTOLOGY_PATH)

    for name in SEC_CRITICAL_CLASSES:
        if not onto.has_name(name):
            raise RuntimeError(f"Sec critical class '{name}' is not in ontology {onto.version}")

    manifest = PackManifest(
        id="sec",
        version=str(_META["version"]),
        display_name=str(_META["display_name"]),
        capabilities=frozenset(_META["capabilities"]),
        default_scene_model=str(_META["default_scene_model"]),
    )

    ontology = OntologySpec(
        yaml_path=_ONTOLOGY_PATH,
        stuff_names=SEC_STUFF_NAMES,
        stuff_l0=frozenset({"surface", "ignore"}),
        supported_core=SEC_SUPPORTED_CORE,
        custom_id_base=CUSTOM_ID_BASE,
    )

    autolabel = AutoLabelProfile(
        vlm_prompt_template=SEC_VLM_PROMPT_TEMPLATE,
        cross_anchors=SEC_CROSS_ANCHORS,
        coco_map=dict(SEC_COCO_TO_ONTOLOGY),
        openvocab_synonyms=dict(SEC_OPENVOCAB_SYNONYMS),
        gate_policy=GatePolicy(safety_l1=SEC_SAFETY_L1, rare_needs_review=True),
        disable_ego_hood_mask=True,  # a fixed camera has no ego bonnet to mask
    )

    eval_strata = EvalStrataSpec(
        dimensions=(
            StratumDimension(name="camera_zone"),
            StratumDimension(name="time_of_day"),
            StratumDimension(name="occupancy"),
            StratumDimension(name="lighting"),
        ),
        protected_slices=("night_lowlight", "crowd_occlusion"),
        critical_class_names=SEC_CRITICAL_CLASSES,
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
        safety_policy=make_safety_policy(
            onto, SEC_SAFETY_L1, SEC_CRITICAL_CLASSES,
            affinity=partial(superclass_affinity_cost, safety_l1=SEC_SAFETY_L1),
        ),
        autolabel_profile=autolabel,
        eval_strata=eval_strata,
        quality_profile=build_quality_profile(),
        forge_targets=(),  # CCTV silicon profiles (NVR SoCs, Ambarella, ARTPEC, x86) land in SEC-M9
        privacy=privacy,
        scene_model=StaticCameraSceneModelFactory(),
        ingestion_adapters=(StaticCameraIngestionAdapter(),),
    )


PACK = _build()
