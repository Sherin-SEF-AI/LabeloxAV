"""VERDYX safety-weighted recall (M15). Not every miss is equal: missing a pedestrian two seconds from
collision is a different event than missing a parked car far away. This weights recall by time-to-collision so
the imminent-hazard misses dominate the number, isolates the critical-object slice (VRUs and cattle, the India
safety classes), and pulls out the near-miss slice a safety reviewer must sign off on. Pure over per-object
records carrying a class, a TTC, and a detected flag, so the weighting is testable."""

from __future__ import annotations


def _default_critical_ids() -> set[int]:
    """The active pack's safety-critical classes, resolved to ids through the governed ontology. Replaces the
    old hardcoded CRITICAL_CLASSES = {0,1,2,3,8}, which was 0-based and disagreed with the 1-based ontology
    ids (latent bug #2 in docs/AV_ASSUMPTIONS.md); the classes are now named (pedestrian, rider, motorcycle,
    cycle, cattle for AV) and resolved correctly."""
    from services.autolabel.ontology import get_ontology
    from services.domain import critical_class_ids

    return critical_class_ids(get_ontology())


def critical_object_recall(objects: list[dict], critical_ids: set[int] | None = None) -> dict:
    """Recall restricted to the safety-critical classes. Each object is {class_id, detected}. `critical_ids`
    defaults to the active pack's critical classes resolved to ids. Returns the recall and the raw counts, so a
    headline mAP cannot hide a VRU regression."""
    crit_ids = _default_critical_ids() if critical_ids is None else critical_ids
    crit = [o for o in objects if o.get("class_id") in crit_ids]
    n = len(crit)
    hit = sum(1 for o in crit if o.get("detected"))
    return {"n_critical": n, "detected": hit, "recall": round(hit / n, 4) if n else None}


def ttc_weighted_recall(objects: list[dict], ttc_horizon_s: float = 6.0) -> dict:
    """Recall weighted so imminent-collision objects count most. Each object is {detected, ttc_s}; an object
    with a smaller time-to-collision gets a larger weight (linear down to zero at ttc_horizon_s, clamped).
    Objects with no TTC take unit weight. A miss on a 1 s-to-impact VRU costs far more than a miss on a distant
    one."""
    num = den = 0.0
    for o in objects:
        ttc = o.get("ttc_s")
        if ttc is None:
            w = 1.0
        else:
            w = max(0.0, (ttc_horizon_s - float(ttc)) / ttc_horizon_s)
        den += w
        if o.get("detected"):
            num += w
    return {"ttc_weighted_recall": round(num / den, 4) if den else None, "weight_mass": round(den, 4)}


def near_miss_slice(objects: list[dict], ttc_s: float = 2.0) -> dict:
    """The near-miss slice: objects within ttc_s of collision. This is the slice a safety reviewer signs off on
    explicitly. Returns the slice recall and the ids of any missed near-miss objects (the escalation list)."""
    near = [o for o in objects if o.get("ttc_s") is not None and float(o["ttc_s"]) <= ttc_s]
    missed = [o.get("object_id") for o in near if not o.get("detected")]
    n = len(near)
    return {"n_near_miss": n, "detected": n - len(missed),
            "recall": round((n - len(missed)) / n, 4) if n else None,
            "missed_object_ids": missed}
