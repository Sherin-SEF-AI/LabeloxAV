"""VERDYX slice-based verdict (M7). Aggregate mAP hides the failures that matter on Indian roads: a challenger
can improve overall while quietly losing pedestrian-at-night. VERDYX evaluates per slice and gates on the
protected slices, so a regression on any protected slice blocks promotion no matter how much aggregate mAP
improved. The promotion decision itself stays in LabeloxAV governance; VERDYX emits the verdict it consumes.

slice_verdict() is pure over two per-slice metric dicts so the gate is unit-testable (the M7 acceptance)."""

from __future__ import annotations


def slice_verdict(champion: dict, challenger: dict, protected_slices: list[str],
                  slice_metric: str = "map", agg_metric: str = "map50",
                  slice_drop_tol: float = 0.02, min_agg_uplift: float = 0.005) -> dict:
    """Decide {promote, reject, needs_review} for a challenger against a champion.

    champion/challenger are {"aggregate": {agg_metric: float}, "per_slice": {slice_id: {slice_metric: float}}}.
    Reject when any protected slice regresses past slice_drop_tol. Otherwise promote when aggregate improves by
    at least min_agg_uplift, else needs_review."""
    ch_slices = challenger.get("per_slice", {})
    cp_slices = champion.get("per_slice", {})
    regressed = []
    for sl in protected_slices:
        cp = (cp_slices.get(sl) or {}).get(slice_metric)
        chal = (ch_slices.get(sl) or {}).get(slice_metric)
        if cp is None:
            continue  # champion never measured this slice; nothing to regress against
        # Fail-closed (C6): a protected slice the champion measured but the challenger DROPPED is a regression,
        # not a silent pass. Omitting the slice must not be a way to bypass the gate.
        if chal is None:
            regressed.append({"slice": sl, "from": round(cp, 4), "to": None,
                              "drop": None, "reason": "unmeasured in challenger"})
        elif chal < cp - slice_drop_tol:
            regressed.append({"slice": sl, "from": round(cp, 4), "to": round(chal, 4),
                              "drop": round(cp - chal, 4)})

    cp_agg = float((champion.get("aggregate") or {}).get(agg_metric, 0.0))
    chal_agg = float((challenger.get("aggregate") or {}).get(agg_metric, 0.0))
    beats = chal_agg >= cp_agg + min_agg_uplift

    reasons: list[str] = []
    if regressed:
        verdict = "reject"
        reasons.append(f"protected slice regression: {[r['slice'] for r in regressed]}")
    elif beats:
        verdict = "promote"
        reasons.append(f"aggregate {agg_metric} {cp_agg:.3f} -> {chal_agg:.3f} with no protected regression")
    else:
        verdict = "needs_review"
        reasons.append(f"no aggregate uplift ({cp_agg:.3f} -> {chal_agg:.3f})")

    return {"verdict": verdict, "regressed_slices": regressed, "beats_aggregate": beats,
            "agg_delta": round(chal_agg - cp_agg, 4), "reasons": reasons}


def slice_matrix(champion: dict, challenger: dict, slices: list[str], slice_metric: str = "map") -> list[dict]:
    """Rows for the slice-matrix UI: per slice, the champion and challenger metric and the regression state
    (colored only by state in the UI: worse = red, better = green, same = neutral)."""
    out = []
    cp, ch = champion.get("per_slice", {}), challenger.get("per_slice", {})
    for sl in slices:
        a = (cp.get(sl) or {}).get(slice_metric)
        b = (ch.get(sl) or {}).get(slice_metric)
        delta = round(b - a, 4) if (a is not None and b is not None) else None
        state = "same" if delta is None or abs(delta) < 1e-6 else ("better" if delta > 0 else "worse")
        out.append({"slice": sl, "champion": a, "challenger": b, "delta": delta, "state": state})
    return out
