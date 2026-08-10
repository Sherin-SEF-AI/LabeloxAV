"""Explainable auto-labeling (M-F.0): the decision story for any object, assembled from the provenance that
the fusion and gate already recorded. This adds NO new inference; it reads provenance.proposals (which paths
proposed the object and their scores), provenance.agreement, the VLM verdict, the quality-reviewer flags, and
the calibrated confidence, then replays the gate's own logic to name the deciding reason. A reviewer sees why
a label was accepted; a buyer's diligence gets a real answer instead of a bare number.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import Object
from services.autolabel.calibrate import calibrate_confidence
from services.autolabel.gate import class_auto_accept, is_rare
from services.autolabel.ontology import get_ontology

_PATH_LABEL = {
    "path_a_yolo26": "YOLO detector",
    "path_b_sam3": "open-vocabulary + SAM",
    "path_c_qwen3vl": "VLM verifier",
}
_QUALITY_LABEL = {
    "above_horizon": "box sits above the horizon (likely sky or background)",
    "impossible_size": "implausible size for its class",
    "duplicate": "duplicate of another object",
    "part_of": "a part contained in a larger object (e.g. a wheel inside a vehicle)",
    "vru_in_vehicle": "a person or rider geometrically inside a vehicle box",
    "tiny": "too small to be a real object",
    "edge_sliver": "a thin sliver at the frame edge",
}


def _human(path: str) -> str:
    return _PATH_LABEL.get(path, path)


def _conf(p: dict) -> float:
    """A proposal's confidence, coercing a missing or null score to 0 (some paths record no scalar conf)."""
    v = p.get("conf")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


async def explain_object(db: AsyncSession, object_id: UUID) -> dict:
    """The full rationale for one object: paths, agreement, VLM, calibration, quality, rarity, conflict, and
    the deciding reason, plus a plain-language summary. Everything is read from stored provenance."""
    obj = await db.get(Object, object_id)
    if obj is None:
        return {"error": "object not found"}
    onto = get_ontology()
    cfg = get_settings()
    prov = obj.provenance or {}

    try:
        cls = onto.by_id(int(obj.class_id))
        class_name = cls.name
    except Exception:  # noqa: BLE001
        class_name = str(obj.class_id)
    rare = False
    try:
        rare = is_rare(int(obj.class_id), onto)
    except Exception:  # noqa: BLE001
        pass

    proposals = prov.get("proposals") or []
    agreement = bool(prov.get("agreement"))
    mask_box_disagree = bool(prov.get("mask_box_disagree"))
    quality_flags = list(prov.get("quality_flags") or [])
    calibrated_from = prov.get("calibrated_from")

    # the calibrated confidence the CURRENT gate would read for this raw score (consistent with the live gate)
    calibrated = None
    if isinstance(calibrated_from, (int, float)) and not isinstance(calibrated_from, bool):
        class_disagree = any(p.get("verdict") == "overruled" for p in proposals) and len(proposals) > 1
        calibrated = round(calibrate_confidence(float(calibrated_from), agreement, class_disagree,
                                                mask_box_disagree, cfg.calibrate), 3)
    threshold = None
    try:
        threshold = round(class_auto_accept(int(obj.class_id), onto, cfg.gate), 3)
    except Exception:  # noqa: BLE001
        pass

    # the VLM proposal, if the object was duty-cycled through Path C
    vlm_prop = next((p for p in proposals if p.get("path") == "path_c_qwen3vl"), None)
    vlm = None
    if vlm_prop is not None:
        vlm = {"confirmed": vlm_prop.get("verdict") in ("confirm", "agree"),
               "class_name": vlm_prop.get("class_name"), "verdict": vlm_prop.get("verdict")}

    overruled = [p for p in proposals if p.get("verdict") == "overruled"]

    # replay the gate to name the deciding reason for the machine's route (independent of any later human edit)
    quality_ok = not quality_flags
    vlm_ok = vlm is not None and vlm["confirmed"]
    decision, reason = _replay_gate(calibrated, threshold, agreement, rare, mask_box_disagree, quality_ok,
                                    vlm_ok, cfg)

    # plain-language summary
    summary: list[str] = []
    named = [p for p in proposals if p.get("path") != "path_c_qwen3vl"]
    if len(named) >= 2:
        parts = ", ".join(f"{_human(p['path'])} at {round(_conf(p), 2)}" for p in named)
        summary.append(f"{len(named)} paths proposed this: {parts}. "
                       + ("They agreed on the class." if agreement else "They did not agree on the class."))
    elif len(named) == 1:
        p = named[0]
        summary.append(f"One path proposed this: {_human(p['path'])} at {round(_conf(p), 2)}. "
                       "There was no cross-path agreement.")
    if overruled:
        oc = ", ".join(sorted({str(p.get("class_name")) for p in overruled}))
        summary.append(f"A conflicting class was overruled by the weighted vote: {oc}.")
    if vlm is not None:
        summary.append(f"The VLM verifier {'confirmed' if vlm['confirmed'] else 'did not confirm'} it"
                       + (f" as '{vlm['class_name']}'." if vlm.get("class_name") else "."))
    else:
        summary.append("The VLM verifier was not consulted for this object.")
    if calibrated is not None and threshold is not None:
        summary.append(f"Calibrated confidence is {calibrated} (raw {round(float(calibrated_from), 2)}); the "
                       f"auto-accept floor for this class is {threshold}.")
    if quality_flags:
        flags = "; ".join(_QUALITY_LABEL.get(f, f) for f in quality_flags)
        summary.append(f"The quality reviewer flagged: {flags}.")
    else:
        summary.append("The quality reviewer found no geometric or contextual problems.")
    if rare:
        summary.append("This is a rare or long-tail class, so it needs cross-path agreement AND a VLM "
                       "confirmation to auto-accept.")
    summary.append(f"Decision: {decision}. {reason}")

    if obj.source == "human":
        summary.append("A human has since annotated or corrected this object.")
    elif prov.get("agent_run_id"):
        summary.append("An agent proposed this change (reversible via its run).")

    return {
        "object_id": str(object_id), "class_name": class_name, "state": obj.state, "source": obj.source,
        "rare": rare,
        "paths": [{"path": p.get("path"), "label": _human(p.get("path", "")), "class_name": p.get("class_name"),
                   "conf": round(_conf(p), 4), "verdict": p.get("verdict"),
                   "model_version": p.get("model_version")} for p in proposals],
        "agreement": agreement, "mask_box_disagree": mask_box_disagree,
        "vlm": vlm, "overruled_classes": sorted({str(p.get("class_name")) for p in overruled}),
        "calibration": {"raw": round(float(calibrated_from), 4)
                        if isinstance(calibrated_from, (int, float)) and not isinstance(calibrated_from, bool) else None,
                        "calibrated": calibrated, "auto_accept_floor": threshold},
        "quality_flags": quality_flags,
        "machine_decision": decision, "deciding_reason": reason,
        "summary": summary,
    }


def _replay_gate(calibrated, threshold, agreement, rare, mask_box_disagree, quality_ok, vlm_ok, cfg) -> tuple[str, str]:
    """Name the machine's route and the one reason that determined it, mirroring gate_object's order."""
    g = cfg.gate
    if calibrated is None:
        return "review", "No single pre-calibration score was recorded, so it could not auto-accept."
    if calibrated < g.review_low:
        return "annotate", (f"Calibrated confidence {calibrated} is below the review floor {round(g.review_low, 3)}, "
                            "so the pre-label was too weak to keep and full annotation is needed.")
    if not quality_ok:
        return "review", "The quality reviewer demoted it, so it cannot auto-accept regardless of confidence."
    if g.force_review_on_mask_box_disagree and mask_box_disagree:
        return "review", "The mask and box disagree on geometry, which forces a human review."
    if threshold is not None and calibrated < threshold:
        return "review", (f"Calibrated confidence {calibrated} is below the auto-accept floor {threshold}, so a "
                          "human reviews it.")
    if not agreement:
        return "review", "The paths did not agree on the class, and agreement is required to auto-accept."
    if rare and g.rare_needs_agreement_and_vlm and not vlm_ok:
        return "review", ("This is a rare class, which needs a VLM confirmation on top of agreement, and the "
                          "VLM did not confirm it.")
    if rare and g.rare_needs_agreement_and_vlm:
        return "auto_accept", "It cleared the floor, the paths agreed, and the VLM confirmed the rare class."
    return "auto_accept", "It cleared the auto-accept floor and the paths agreed on the class."
