"""Diff proposed vs existing labels and classify each change (M4.2). The verdict drives selective apply:

  unchanged    - same class, nothing to do
  conflict     - touches a human-verified object; never auto-apply, route to review
  improvement  - an ontology promotion, or a confident class upgrade; safe to auto-apply
  regression   - the new class drops confidence sharply; flag and route to review
  review       - a low-confidence change; route to review

Pure function over a proposal and the configured thresholds.
"""

from __future__ import annotations

from core.config import get_settings

_HUMAN_SOURCES = ("human",)


def classify_change(proposal: dict) -> dict:
    cfg = get_settings().phase4.relabel
    old_c, new_c = proposal["old_class_id"], proposal["new_class_id"]
    old_conf, new_conf = float(proposal.get("old_conf", 0.0)), float(proposal.get("new_conf", 0.0))
    is_human = proposal.get("source") in cfg.never_touch_sources or proposal.get("source") in _HUMAN_SOURCES
    promotion = proposal.get("reason") == "ontology_promotion"

    if old_c == new_c and not promotion:
        return {"verdict": "unchanged", "apply": False, "reason": "same class"}
    if is_human:
        return {"verdict": "conflict", "apply": False, "reason": "human-verified, never overwritten"}
    if promotion:
        return {"verdict": "improvement", "apply": True, "reason": "ontology promotion of a fallback class"}
    if new_conf >= cfg.auto_apply_min_conf and new_conf >= old_conf + cfg.auto_apply_min_uplift:
        return {"verdict": "improvement", "apply": True,
                "reason": f"confident upgrade ({old_conf:.2f} -> {new_conf:.2f})"}
    if new_conf < old_conf - cfg.regression_margin:
        return {"verdict": "regression", "apply": False, "reason": f"confidence drop ({old_conf:.2f} -> {new_conf:.2f})"}
    return {"verdict": "review", "apply": False, "reason": "low-confidence change"}


def summarize(classified: list[dict]) -> dict:
    out: dict = {}
    for c in classified:
        out[c["verdict"]] = out.get(c["verdict"], 0) + 1
    return out
