"""The confidence gate: where humans enter (Principle 04). Calibrated confidence routes each object to
auto_accept, review, or annotate.

M-Q.4 hardening:
  - Per-class calibrated thresholds replace the global constant: safety-critical classes (VRU, animal)
    must be near-certain (0.99), benign classes use the default (0.95). Confidence is calibrated, so a
    threshold is a precision floor.
  - A rare/fallback class earns auto-accept only with cross-path agreement AND VLM confirmation, never on
    one model's output. This kills confident-but-wrong rare detections.
  - The quality reviewer's verdict (geometric/contextual nonsense) demotes an object before it can
    auto-accept, regardless of score.
"""

from __future__ import annotations

from core.config import GateSettings
from core.schemas import GateState, Provenance, UnifiedObject
from services.autolabel.ontology import Ontology

_SAFETY_L1 = {"vru", "animal"}


def is_rare(class_id: int, onto: Ontology) -> bool:
    c = onto.by_id(class_id)
    return c.india or c.l1 == "fallback"


def class_auto_accept(class_id: int, onto: Ontology, cfg: GateSettings) -> float:
    """Per-class auto-accept threshold: safety-critical classes near-certain, benign classes the default."""
    return cfg.safety_auto_accept if onto.by_id(class_id).l1 in _SAFETY_L1 else cfg.auto_accept


def vlm_confirmed(prov: Provenance) -> bool:
    """The VLM saw this object and confirmed (did not overrule) its class."""
    return any(p.path == "path_c_qwen3vl" and p.verdict in ("confirm", "agree") for p in prov.proposals)


def gate_object(obj: UnifiedObject, onto: Ontology, cfg: GateSettings,
                auto_accept_enabled: bool = True, quality_ok: bool = True) -> GateState:
    conf = obj.conf
    prov = obj.provenance
    rare = is_rare(obj.class_id, onto)

    # Below the review floor is always a full annotate, whatever else is true.
    if conf < cfg.review_low:
        return GateState.annotate

    # The quality reviewer demoted geometric/contextual nonsense (sky box, impossible size, tyre-as-vehicle,
    # duplicate, pedestrian-in-car). It never auto-accepts; a human confirms or kills it.
    if not quality_ok:
        return GateState.review

    if cfg.force_review_on_mask_box_disagree and prov.mask_box_disagree:
        return GateState.review

    # Strict escape hatch: when set, a rare/fallback class never auto-accepts, whatever else is true. Off by
    # default because M-Q.4's agreement+VLM rule below is the smarter policy; flip on to fully freeze the
    # long tail (e.g. a fresh ontology before any rare class has earned trust).
    if cfg.force_review_on_rare and rare:
        return GateState.review

    # auto_accept_enabled is the governance kill switch: when the loop is paused, nothing auto-accepts.
    if not auto_accept_enabled:
        return GateState.review

    # Per-class calibrated threshold plus cross-path agreement are the baseline for any auto-accept.
    if conf < class_auto_accept(obj.class_id, onto, cfg) or not prov.agreement:
        return GateState.review

    # A rare/fallback class must also be VLM-confirmed: agreement alone is not enough for the long tail.
    if rare and cfg.rare_needs_agreement_and_vlm and not vlm_confirmed(prov):
        return GateState.review

    return GateState.auto_accept


def needs_vlm(obj: UnifiedObject, onto: Ontology, cfg: GateSettings, quality_ok: bool = True) -> bool:
    """Path C (VLM) duty-cycle predicate. True only for the uncertain subset: paths disagree, confidence in
    the (per-class) review band, a rare/fallback class, a mask conflict, or a quality-flagged object that a
    second look should confirm or kill. Never the full stream."""
    prov = obj.provenance
    class_disagree = any(p.verdict == "overruled" for p in prov.proposals) and len(prov.proposals) > 1
    in_review_band = cfg.review_low <= obj.conf < class_auto_accept(obj.class_id, onto, cfg)
    return bool(class_disagree or in_review_band or is_rare(obj.class_id, onto)
                or prov.mask_box_disagree or not quality_ok)
