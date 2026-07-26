"""FORGYX model-target co-optimization (M16). A model and its silicon are optimized together, not in isolation:
the pruning ratio and quantization that fit an AGX Orin's latency budget differ from a Pi+Hailo's. This plans
the co-optimization, estimates the latency and accuracy an optimization step trades, and ranks the candidate
(prune, quantize) configurations by whether they clear the target's latency budget at the least accuracy cost.
The actual prune/quantize/compile runs stay in services/forgyx/optimize (capability-gated to where the backend
exists); this is the planner that decides which configuration to build, so it is pure and testable."""

from __future__ import annotations

# per-target latency budgets (ms, p95), keyed by the deployment target. Legacy AV silicon registry; the
# active-pack forge targets are consulted first, and this is the fallback for targets a pack does not declare.
TARGET_BUDGET_MS = {
    "sentrixai_litert": 33.0,   # mobile, ~30 fps
    "agx_orin_trt": 15.0,       # high-end automotive
    "orin_nano_trt": 40.0,      # mid automotive
    "pi_hailo": 50.0,           # low-power edge
}


def _budget_for(target: str) -> float | None:
    """The target's latency budget from any pack's forge targets, else the legacy AV registry. For an AV target
    the pack value mirrors the registry, so the result is identical."""
    from services.domain import resolve_forge_target

    ft = resolve_forge_target(target)
    if ft is not None:
        return ft.latency_budget_ms
    return TARGET_BUDGET_MS.get(target)


def estimate_step(base_latency_ms: float, base_map50: float, prune: float, int8: bool) -> dict:
    """Estimate the latency and accuracy of one (prune ratio, int8) configuration from a base fp16 model.

    Pruning a fraction ``prune`` of channels reduces latency roughly linearly and costs accuracy super-linearly
    (the last channels matter most); int8 quantization gives a further latency cut at a small accuracy cost.
    These are planning estimates; the built artifact is always re-benchmarked and re-evaluated before it can
    deploy."""
    prune = max(0.0, min(0.9, prune))
    lat = base_latency_ms * (1.0 - 0.85 * prune)
    map50 = base_map50 * (1.0 - 0.5 * prune * prune)
    if int8:
        lat *= 0.6
        map50 *= 0.985
    return {"prune": round(prune, 3), "int8": int8, "est_latency_ms": round(lat, 3),
            "est_map50": round(map50, 4)}


def plan_cooptimization(target: str, base_latency_ms: float, base_map50: float,
                        prune_grid: list[float] | None = None) -> dict:
    """Rank co-optimization configurations for a target: keep only those whose estimated latency clears the
    target budget, then pick the one that clears it at the highest estimated accuracy. Returns the chosen
    configuration and the full ranked grid; feasible=False when nothing in the grid meets the budget (a signal
    to widen the grid or pick a smaller architecture)."""
    budget = _budget_for(target)
    if budget is None:
        raise ValueError(f"unknown target {target}")
    grid = prune_grid if prune_grid is not None else [0.0, 0.2, 0.4, 0.6]
    cands = [estimate_step(base_latency_ms, base_map50, p, q) for p in grid for q in (False, True)]
    feasible = [c for c in cands if c["est_latency_ms"] <= budget]
    feasible.sort(key=lambda c: (-c["est_map50"], c["est_latency_ms"]))
    chosen = feasible[0] if feasible else None
    return {"target": target, "budget_ms": budget, "feasible": bool(feasible), "chosen": chosen,
            "ranked": sorted(cands, key=lambda c: c["est_latency_ms"])}
