"""Adaptive collection tasking (M17). A label budget only helps if the data exists; when SIEVYX reports an ODD
cell that is starved or entirely missing, no amount of labeling fixes it, only collection does. This turns ODD
coverage gaps into ranked collection tasks with a target sample count to close the gap, prioritized by the size
of the deficit and the safety weight of the cell. Pure over the gap records SIEVYX odd.coverage_gaps produces,
so the target counts and ranking are testable."""

from __future__ import annotations

# ODD cells whose scenario keys mark a safety-critical condition get a collection priority boost
SAFETY_KEYWORDS = ("vru", "pedestrian", "rider", "cattle", "night", "rain", "junction", "unmarked")


def _safety_weight(cell: str) -> float:
    key = cell.lower()
    hits = sum(1 for kw in SAFETY_KEYWORDS if kw in key)
    return 1.0 + 0.5 * hits


def collection_tasks(odd_gaps: list[dict], total_samples: int) -> list[dict]:
    """Turn ODD coverage gaps into ranked collection tasks.

    Each gap is {cell, gap, target_frac, actual_frac, count, missing} (the SIEVYX shape). The target sample
    count to close a gap is gap * total_samples rounded up (how many more samples that cell needs to reach its
    target fraction of the current fleet size). Priority is the deficit scaled by the cell's safety weight, so a
    missing night-rain-junction cell outranks a mildly-underrepresented benign cell. Returns tasks highest
    priority first."""
    import math

    tasks = []
    for g in odd_gaps:
        cell = g.get("cell", "")
        gap = max(0.0, float(g.get("gap", 0.0)))
        sw = _safety_weight(cell)
        target_count = int(math.ceil(gap * max(total_samples, 0)))
        priority = round(gap * sw, 4)
        tasks.append({"cell": cell, "priority": priority, "target_count": target_count,
                      "missing": bool(g.get("missing", False)), "safety_weight": round(sw, 3),
                      "reason": f"ODD gap {gap:.3f} on {cell}" + (" (missing)" if g.get("missing") else "")})
    tasks.sort(key=lambda t: t["priority"], reverse=True)
    return tasks
