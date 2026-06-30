"""Recall promotion gate (pure). Recall is the metric the recovery layer exists to fix, so a challenger
must prove it on the safety classes or it is refused, exactly as the Safe-mIoU check refuses a model
that cannot prove it did not regress safety. Fail-closed: a challenger that cannot report per-class
recall for the safety classes (VRU, animal) is not promoted.

These functions are pure over the gold-metric dicts and the ontology; champion_gate folds them into the
promotion decision.
"""

from __future__ import annotations

_VRU_ANIMAL = {"vru", "animal"}


def _l1(name: str, onto) -> str | None:
    try:
        return onto.by_name(name).l1
    except Exception:  # noqa: BLE001  (a metric key that is not an ontology class is simply not a safety class)
        return None


def _is_safety(name: str, onto) -> bool:
    return _l1(name, onto) in _VRU_ANIMAL


def resolve_safety_drop(class_name: str, onto, cfg) -> float:
    """The per-class safety-AP drop tolerance. cfg.safety_class_drop may be a float (global, old
    behavior) or a dict keyed by L1 with a _default; VRU and animal are held tighter than the global
    fallback."""
    drop = cfg.safety_class_drop
    if isinstance(drop, int | float):
        return float(drop)
    l1 = _l1(class_name, onto)
    if l1 in drop:
        return float(drop[l1])
    return float(drop.get("_default", 0.15))


def safety_recall_floor(challenger: dict, onto, cfg) -> dict:
    """Block when any safety class recall is below cfg.safety_recall_floor. Fail-closed: if
    require_safety_recall and the challenger reports no per_class_recall, refuse."""
    pcr = challenger.get("per_class_recall")
    if pcr is None:
        if getattr(cfg, "require_safety_recall", True):
            return {"ok": False, "reasons": ["no per_class_recall reported (fail-closed)"],
                    "below_floor": [], "regressed": []}
        return {"ok": True, "reasons": [], "below_floor": [], "regressed": []}
    floor = float(cfg.safety_recall_floor)
    below = sorted(cn for cn, r in pcr.items() if _is_safety(cn, onto) and float(r) < floor)
    ok = not below
    reasons = [] if ok else [f"safety-class recall below floor {floor}: {below}"]
    return {"ok": ok, "reasons": reasons, "below_floor": below, "regressed": []}


def safety_recall_no_regress(challenger: dict, champion: dict | None, onto, cfg) -> dict:
    """Even above the floor, a safety class must not lose more than cfg.safety_recall_max_drop against
    the incumbent. Skipped when no incumbent recall baseline exists."""
    chal = challenger.get("per_class_recall") or {}
    champ = (champion or {}).get("per_class_recall") or {}
    if not champ:
        return {"ok": True, "reasons": [], "below_floor": [], "regressed": []}
    max_drop = float(cfg.safety_recall_max_drop)
    regressed = sorted(cn for cn, base in champ.items()
                       if _is_safety(cn, onto) and cn in chal and float(chal[cn]) < float(base) - max_drop)
    ok = not regressed
    reasons = [] if ok else [f"safety-class recall regressed beyond {max_drop}: {regressed}"]
    return {"ok": ok, "reasons": reasons, "below_floor": [], "regressed": regressed}
