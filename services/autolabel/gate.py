"""The confidence gate: where humans enter (Principle 04). Calibrated confidence routes each
object to auto_accept, review, or annotate. Mask-vs-box disagreement or rare/fallback membership
force a review regardless of score. This keeps annotation cost sublinear in data volume.

Thresholds are config values, intended to become per-class and learned once a gold set exists.
"""

from __future__ import annotations

from core.config import GateSettings
from core.schemas import GateState, UnifiedObject
from services.autolabel.ontology import Ontology


def is_rare(class_id: int, onto: Ontology) -> bool:
    c = onto.by_id(class_id)
    return c.india or c.l1 == "fallback"


def gate_object(obj: UnifiedObject, onto: Ontology, cfg: GateSettings,
                auto_accept_enabled: bool = True) -> GateState:
    conf = obj.conf
    prov = obj.provenance
    rare = is_rare(obj.class_id, onto)

    # Below the review floor is always a full annotate, whatever else is true.
    if conf < cfg.review_low:
        return GateState.annotate

    # Forced reviews: a rare/fallback class or a geometry conflict never auto-accepts.
    forced_review = (cfg.force_review_on_rare and rare) or (
        cfg.force_review_on_mask_box_disagree and prov.mask_box_disagree
    )

    # auto_accept_enabled is the governance kill switch: when the loop is paused, nothing auto-accepts
    # and everything falls to review until an operator releases it.
    if auto_accept_enabled and conf >= cfg.auto_accept and prov.agreement and not forced_review:
        return GateState.auto_accept

    return GateState.review


def needs_vlm(obj: UnifiedObject, onto: Ontology, cfg: GateSettings) -> bool:
    """Path C (VLM) duty-cycle predicate (M4). True only for the uncertain subset: paths disagree,
    confidence in the review band, or a rare/fallback class. Never the full stream."""
    prov = obj.provenance
    class_disagree = any(p.verdict == "overruled" for p in prov.proposals) and len(prov.proposals) > 1
    in_review_band = cfg.review_low <= obj.conf < cfg.auto_accept
    return bool(class_disagree or in_review_band or is_rare(obj.class_id, onto) or prov.mask_box_disagree)
