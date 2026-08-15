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
    # Auto-promote only on an unambiguous accuracy PASS ("promote"), never on "needs_review": a compiled model
    # whose slices are unmeasured or show no uplift is a human-review case, not an automatic ship. Treating
    # needs_review as promotable (the old `!= "reject"`) let an under-verified quant slip through.
    if lat["regressed"] or acc["verdict"] == "reject":
        verdict = "block"
    elif acc["verdict"] == "promote":
        verdict = "promote"
    else:
        verdict = "needs_review"
        reasons += acc["reasons"]
    promote = verdict == "promote"
    return {"promote": promote, "verdict": verdict, "latency": lat, "accuracy": acc,
            "reasons": reasons or ["passes: faster or equal, no protected-slice regression"]}


def pareto_rank(benchmarks: list[dict], latency_key: str = "p95", accuracy_key: str = "map50") -> list[dict]:
    """Rank (model, target) benchmarks by Pareto dominance on latency (lower better) against accuracy
    (higher better). rank 0 is the Pareto front.

    A benchmark with no accuracy is not ranked at all, rather than ranked as accuracy zero. The two are very
    different claims: zero says the model was measured and is useless, while absent says nobody has scored it
    on this target yet, and a Pareto plot that conflates them recommends against a model on evidence that
    does not exist. Unranked rows still come back, marked, and sort last, because dropping them would hide a
    real artifact from the page that lists artifacts.

    This also stopped being a hypothetical the moment the first honest benchmark landed. Every row here used
    to be seeded demo data that carried an accuracy reference, so `float(None)` was unreachable; the first
    real export, which has no accuracy because nothing has scored it, made the endpoint 500.
    """
    def lat(b) -> float:
        v = (b.get("latency_ms") or {}).get(latency_key)
        return float(v) if v is not None else float("inf")

    def acc(b) -> float | None:
        v = b.get(accuracy_key)
        return float(v) if v is not None else None

    scored = [b for b in benchmarks if acc(b) is not None]
    out = []
    for b in benchmarks:
        a = acc(b)
        if a is None:
            out.append({**b, "pareto_rank": None, "unranked": True,
                        "unranked_reason": f"no {accuracy_key} measured for this benchmark"})
            continue
        dominated_by = sum(
            1 for o in scored
            if o is not b and lat(o) <= lat(b) and (acc(o) or 0.0) >= a
            and (lat(o) < lat(b) or (acc(o) or 0.0) > a))
        out.append({**b, "pareto_rank": dominated_by, "unranked": False})
    # Unranked last, then by rank, then by latency. `None` cannot be compared to an int, so the flag carries
    # the ordering rather than the rank itself.
    return sorted(out, key=lambda x: (x["pareto_rank"] is None, x["pareto_rank"] or 0, lat(x)))
