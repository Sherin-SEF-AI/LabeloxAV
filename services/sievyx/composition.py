"""SIEVYX label-queue priority and composition. Label budget is the true bottleneck and the Indian long tail
is enormous and mostly redundant, so SIEVYX decides what is worth labeling: it fuses model uncertainty with
embedding-space rarity into one priority score (the active-learning selector already computes this combined
value), and it reports what the budget is currently being spent on so a human can see whether the queue is
dominated by common classes or is actually surfacing the rare tail.

compose() is a pure aggregator over scored items so the composition dashboard is unit-testable; the router
feeds it live scored candidates from services.activelearn.selector."""

from __future__ import annotations


def _rarity_band(r: float) -> str:
    return "high" if r >= 0.66 else ("medium" if r >= 0.33 else "low")


def compose(items: list[dict], top_n: int = 500) -> dict:
    """Summarize what the top priority items are composed of: the class mix, the rarity-band split, and the
    mean per-signal contribution, so the queue-composition dashboard answers "is the budget buying rare data
    or common data". items are active-learning scored candidates (each with value, class_name, scores)."""
    ranked = sorted(items, key=lambda x: x.get("value", 0.0), reverse=True)[:top_n]
    n = len(ranked)
    if n == 0:
        return {"n": 0, "by_class": [], "by_rarity_band": {}, "mean_signals": {}, "mean_value": 0.0}

    by_class: dict[str, dict] = {}
    band = {"low": 0, "medium": 0, "high": 0}
    sig_totals: dict[str, float] = {}
    val_total = 0.0
    for it in ranked:
        cls = it.get("class_name", "unknown")
        c = by_class.setdefault(cls, {"class_name": cls, "count": 0, "value_sum": 0.0})
        c["count"] += 1
        c["value_sum"] += float(it.get("value", 0.0))
        scores = it.get("scores", {})
        band[_rarity_band(float(scores.get("rarity", 0.0)))] += 1
        for k, v in scores.items():
            sig_totals[k] = sig_totals.get(k, 0.0) + float(v)
        val_total += float(it.get("value", 0.0))

    by_class_list = sorted(
        ({"class_name": c["class_name"], "count": c["count"],
          "share": round(c["count"] / n, 4), "mean_value": round(c["value_sum"] / c["count"], 4)}
         for c in by_class.values()),
        key=lambda x: x["count"], reverse=True)
    return {
        "n": n,
        "by_class": by_class_list,
        "by_rarity_band": {k: {"count": v, "share": round(v / n, 4)} for k, v in band.items()},
        "mean_signals": {k: round(v / n, 4) for k, v in sig_totals.items()},
        "mean_value": round(val_total / n, 5),
    }
