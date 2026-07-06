"""Adaptive label-budget allocator (M17). A finite human label budget should flow to where it moves the model
most: the slices VERDYX found regressing and the ODD cells SIEVYX found starved, weighted so a safety-critical
slice is never starved. This turns a set of weighted demands into an integer allocation that sums exactly to
the budget (largest-remainder apportionment), with a guaranteed floor for safety-critical demands. Pure over
the demands, so the apportionment and the safety floor are testable and byte-reproducible."""

from __future__ import annotations


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Apportion ``total`` integer units across weights so the result sums exactly to total (Hamilton method)."""
    wsum = sum(weights)
    if wsum <= 0 or total <= 0:
        return [0] * len(weights)
    quotas = [w / wsum * total for w in weights]
    floors = [int(q) for q in quotas]
    remainder = total - sum(floors)
    # hand the leftover units to the largest fractional remainders
    order = sorted(range(len(weights)), key=lambda i: quotas[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def allocate_label_budget(demands: list[dict], total_budget: int, safety_floor: int = 0) -> list[dict]:
    """Allocate an integer label budget across weighted demands.

    Each demand is {slice, weight, safety_weight?, reason?}; the effective weight is weight * safety_weight
    (default safety_weight 1.0). Safety-critical demands (safety_weight > 1) are guaranteed at least
    ``safety_floor`` labels before the rest is apportioned by effective weight. The returned allocation sums
    exactly to total_budget (or less only when there are fewer demands than the floor would require)."""
    if not demands or total_budget <= 0:
        return []

    eff = [max(0.0, d.get("weight", 0.0)) * max(0.0, d.get("safety_weight", 1.0)) for d in demands]
    # reserve the safety floor for safety-critical demands, capped so it cannot exceed the budget
    critical = [i for i, d in enumerate(demands) if d.get("safety_weight", 1.0) > 1.0]
    reserved = min(len(critical) * safety_floor, total_budget)
    floors = {i: safety_floor for i in critical} if reserved == len(critical) * safety_floor else {}

    remaining = total_budget - sum(floors.values())
    apportioned = _largest_remainder(eff, remaining)

    out = []
    for i, d in enumerate(demands):
        labels = floors.get(i, 0) + apportioned[i]
        out.append({"slice": d.get("slice"), "labels": labels, "weight": round(eff[i], 4),
                    "reason": d.get("reason", "")})
    out.sort(key=lambda x: x["labels"], reverse=True)
    return out
