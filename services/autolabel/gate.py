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

from collections.abc import Mapping

from core.config import GateSettings
from core.logging import get_logger
from core.schemas import GateState, Provenance, UnifiedObject
from services.autolabel.ontology import Ontology

log = get_logger("autolabel_gate")

# Classes already warned about in this process. The warning is worth saying once per class and worth
# nothing tens of thousands of times, which is how often a gate decision happens in one run.
_WARNED_UNFITTED: set[int] = set()


def is_rare(class_id: int, onto: Ontology) -> bool:
    c = onto.by_id(class_id)
    return c.india or c.l1 == "fallback"


def class_auto_accept(class_id: int, onto: Ontology, cfg: GateSettings,
                      fitted: Mapping[int, float] | None = None) -> float:
    """Per-class auto-accept threshold, preferring one that was measured.

    `fitted` is the active ThresholdFit for the model doing the labelling, resolved once at the top of a
    run by services/oraclyx/threshold_fit.py and passed down, so this stays a pure function on a hot path.

    Without a fitted value for the class, this falls back to the configured constant, and says so. The
    fallback is the normal case today and will be for a while, which is exactly why it is logged rather
    than taken silently: a constant that nobody measured is not a precision floor, and an engine that
    cannot tell you which of its thresholds were measured cannot tell you what its accept rate means.

    The warning is per class and per process, not per object. A gate decision happens tens of thousands of
    times a run, and a line each would bury the signal it exists to give.
    """
    from services.domain import safety_l1

    if fitted is not None:
        t = fitted.get(class_id)
        if t is not None:
            return float(t)
    default = cfg.safety_auto_accept if onto.by_id(class_id).l1 in safety_l1() else cfg.auto_accept
    if class_id not in _WARNED_UNFITTED:
        _WARNED_UNFITTED.add(class_id)
        log.warning("gate.threshold_unfitted", class_id=class_id, threshold=default,
                    had_fit=fitted is not None,
                    detail=("no measured operating point for this class; using the configured constant, "
                            "which is not a precision floor because nobody measured the precision at it"))
    return default


def vlm_confirmed(prov: Provenance) -> bool:
    """The VLM saw this object and confirmed (did not overrule) its class."""
    return any(p.path == "path_c_qwen3vl" and p.verdict in ("confirm", "agree") for p in prov.proposals)


def gate_object(obj: UnifiedObject, onto: Ontology, cfg: GateSettings,
                auto_accept_enabled: bool = True, quality_ok: bool = True,
                fitted: Mapping[int, float] | None = None) -> GateState:
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
    if conf < class_auto_accept(obj.class_id, onto, cfg, fitted) or not prov.agreement:
        return GateState.review

    # A rare/fallback class must also be VLM-confirmed: agreement alone is not enough for the long tail.
    if rare and cfg.rare_needs_agreement_and_vlm and not vlm_confirmed(prov):
        return GateState.review

    return GateState.auto_accept


def needs_vlm(obj: UnifiedObject, onto: Ontology, cfg: GateSettings, quality_ok: bool = True,
              fitted: Mapping[int, float] | None = None) -> bool:
    """Path C (VLM) duty-cycle predicate. True only for the uncertain subset: paths disagree, confidence in
    the (per-class) review band, a rare/fallback class, a mask conflict, or a quality-flagged object that a
    second look should confirm or kill. Never the full stream."""
    prov = obj.provenance
    class_disagree = any(p.verdict == "overruled" for p in prov.proposals) and len(prov.proposals) > 1
    in_review_band = cfg.review_low <= obj.conf < class_auto_accept(obj.class_id, onto, cfg, fitted)
    return bool(class_disagree or in_review_band or is_rare(obj.class_id, onto)
                or prov.mask_box_disagree or not quality_ok)
