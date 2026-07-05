"""FORGYX dual regression gate (M8). The trap pure latency benchmarking misses: a model can get faster and
quietly lose the night-pedestrian slice. So FORGYX blocks promotion on latency regression AND on accuracy
regression, re-verifying the quantized/compiled model through the VERDYX slices. Pure, so the "faster but
drops a protected slice is blocked" acceptance is deterministic."""

from __future__ import annotations

from services.verdyx.verdict import slice_verdict


def latency_regressed(baseline_ms: dict, candidate_ms: dict, pct_tol: float = 0.10, key: str = "p95") -> dict:
    """A candidate must not be slower than the baseline by more than pct_tol at the chosen percentile."""
    b = float((baseline_ms or {}).get(key, 0.0))
    c = float((candidate_ms or {}).get(key, 0.0))
    if b <= 0:
        return {"regressed": False, "baseline": b, "candidate": c, "delta_pct": None}
    delta = (c - b) / b
    return {"regressed": delta > pct_tol, "baseline": b, "candidate": c, "delta_pct": round(delta, 4),
            "faster": c < b}


def dual_gate(baseline_latency: dict, candidate_latency: dict,
              champion_eval: dict, candidate_eval: dict, protected_slices: list[str],
              pct_tol: float = 0.10) -> dict:
    """Block on latency OR accuracy regression. candidate_eval / champion_eval are the VERDYX per-slice evals
    of the compiled model (re-verified) and the champion. Returns promote/block with the reasons."""
    lat = latency_regressed(baseline_latency, candidate_latency, pct_tol)
    acc = slice_verdict(champion_eval, candidate_eval, protected_slices)

    reasons: list[str] = []
    if lat["regressed"]:
        reasons.append(f"latency regressed {lat['delta_pct']:.1%} at p95 (> {pct_tol:.0%})")
    if acc["verdict"] == "reject":
        reasons += acc["reasons"]
    promote = not lat["regressed"] and acc["verdict"] != "reject"
    return {"promote": promote, "latency": lat, "accuracy": acc,
            "reasons": reasons or ["passes: faster or equal, no protected-slice regression"]}


def pareto_rank(benchmarks: list[dict], latency_key: str = "p95", accuracy_key: str = "map50") -> list[dict]:
    """Rank (model, target) benchmarks by Pareto dominance on latency (lower better) vs accuracy (higher
    better). rank 0 is the Pareto front. Each item needs latency_ms[latency_key] and accuracy_ref_map."""
    def lat(b):
        return float((b.get("latency_ms") or {}).get(latency_key, float("inf")))

    def acc(b):
        return float(b.get(accuracy_key, 0.0))

    out = []
    for b in benchmarks:
        dominated_by = sum(
            1 for o in benchmarks
            if o is not b and lat(o) <= lat(b) and acc(o) >= acc(b)
            and (lat(o) < lat(b) or acc(o) > acc(b)))
        out.append({**b, "pareto_rank": dominated_by})
    return sorted(out, key=lambda x: (x["pareto_rank"], lat(x)))
