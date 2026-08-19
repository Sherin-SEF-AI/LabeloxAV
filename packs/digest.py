"""A deterministic, serialisable digest of a domain pack.

This is the substance the golden regression harness freezes (tests/test_golden_av_pack.py): a stable JSON
snapshot of every pack surface. Any change to a pack's ontology, safety definition, auto-label profile, eval
strata, quality profile, forge targets, or privacy plane changes the digest and trips the golden test. It is
the permanent AV parity gate the mission requires, and it works for every future pack too.

The ontology part hashes only governed classes (id < custom_id_base): annotator-added custom classes live in
an environment-specific sidecar and must not make the digest machine-dependent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from packs.registry import get_pack
from services.autolabel.ontology import get_ontology


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pack_digest(pack_id: str = "av") -> dict:
    """Deterministic snapshot of a pack. Sorted throughout so it is stable across runs and machines."""
    p = get_pack(pack_id)
    onto = get_ontology(pack_id)
    base = p.ontology.custom_id_base

    governed = sorted(
        (c.id, c.name, c.l0, c.l1, c.india) for c in onto.classes if c.id < base
    )
    safety_class_names = sorted(
        c.name for c in onto.classes if c.id < base and p.safety_policy.is_safety_class(c.name)
    )

    return {
        "manifest": {
            "id": p.manifest.id,
            "version": p.manifest.version,
            "display_name": p.manifest.display_name,
            "capabilities": sorted(p.manifest.capabilities),
            "default_scene_model": p.manifest.default_scene_model,
        },
        "ontology": {
            "version": onto.version,
            "n_governed_classes": len(governed),
            "governed_classes_sha256": _sha256(governed),
            "stuff_names": sorted(p.ontology.stuff_names),
            "stuff_l0": sorted(p.ontology.stuff_l0),
            "supported_core": sorted(p.ontology.supported_core),
            "custom_id_base": base,
            "fallback_ids": sorted(onto.fallback_ids()),
        },
        "safety": {
            "safety_l1": sorted(p.autolabel_profile.gate_policy.safety_l1),
            "safety_class_names": safety_class_names,
            "critical_class_names": sorted(p.safety_policy.critical_class_names()),
            # The auto-accept false-accept bounds, snapshotted per governed class rather than as the tier
            # table they come from. The tiers are an implementation of the policy; what a reader needs
            # pinned is the bound each class actually gets, because that is what a fitted threshold is
            # solved against and a retiering that silently moved a class would not show up otherwise.
            "accept_far_bounds": {c.name: p.safety_policy.accept_far_bound(c.name)
                                  for c in sorted(onto.classes, key=lambda c: c.name) if c.id < base},
        },
        "autolabel": {
            "vlm_prompt_template": p.autolabel_profile.vlm_prompt_template,
            "cross_anchors": list(p.autolabel_profile.cross_anchors),
            "coco_map": dict(sorted(p.autolabel_profile.coco_map.items())),
            "openvocab_synonyms": {
                k: list(v) for k, v in sorted(p.autolabel_profile.openvocab_synonyms.items())
            },
            "disable_ego_hood_mask": p.autolabel_profile.disable_ego_hood_mask,
        },
        "eval_strata": {
            "dimensions": [d.name for d in p.eval_strata.dimensions],
            "protected_slices": list(p.eval_strata.protected_slices),
            "critical_class_names": sorted(p.eval_strata.critical_class_names),
        },
        "quality_profile": {
            "image_checks": list(p.quality_profile.image_checks),
            "sensor_checks": [
                {"name": s.name, "params": dict(sorted(s.params.items()))}
                for s in p.quality_profile.sensor_checks
            ],
            "rootcause_signatures": [s.name for s in p.quality_profile.rootcause_signatures],
        },
        "forge_targets": [
            {
                "name": t.name,
                "backend": t.backend,
                "backend_modules": list(t.backend_modules),
                "throttle_temp_c": t.throttle_temp_c,
                "power_ceiling_w": t.power_ceiling_w,
                "latency_budget_ms": t.latency_budget_ms,
                "export_format": t.export_format,
            }
            for t in sorted(p.forge_targets, key=lambda t: t.name)
        ],
        "privacy": {
            "redaction_targets": [
                {"name": r.name, "detector": r.detector} for r in p.privacy.redaction_targets
            ],
            "legal_regime": p.privacy.legal_regime,
        },
        # SEC-M1: scene_model / ingestion adapters land in SEC-M2. The golden records their absence so the
        # milestone that adds them updates the frozen snapshot deliberately.
        "scene_model_present": p.scene_model is not None,
        "n_ingestion_adapters": len(p.ingestion_adapters),
    }
