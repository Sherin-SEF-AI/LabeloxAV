"""Label-budget efficiency report (M19). The flywheel spends a finite label budget across slices; this measures
what each slice's labels actually bought in model accuracy, so the next cycle can shift budget toward the
slices with the best return and away from spend that did not move the model. It reports accuracy gain per
thousand labels per slice, ranks slices by that return, and flags negative-ROI spend (labels that hurt, the
unreviewed-pseudo-label failure mode). Pure over the per-slice records, so the ROI math and ranking are
testable."""

from __future__ import annotations


def slice_efficiency(entries: list[dict]) -> list[dict]:
    """Per-slice label ROI. Each entry is {slice, labels, map_before, map_after}. Returns the accuracy gain and
    the gain per 1000 labels, with a negative_roi flag when the slice lost accuracy despite spending labels."""
    out = []
    for e in entries:
        labels = max(0, int(e.get("labels", 0)))
        gain = float(e.get("map_after", 0.0)) - float(e.get("map_before", 0.0))
        per_k = (gain / labels * 1000.0) if labels else 0.0
        out.append({"slice": e.get("slice"), "labels": labels, "gain": round(gain, 4),
                    "gain_per_1k": round(per_k, 4), "negative_roi": labels > 0 and gain < 0})
    out.sort(key=lambda r: r["gain_per_1k"], reverse=True)
    return out


def efficiency_report(entries: list[dict]) -> dict:
    """The label-budget efficiency report over a cycle: per-slice ROI ranked best-first, the totals, the overall
    gain per 1000 labels, and the slices that returned nothing or lost accuracy (the budget to reclaim next
    cycle)."""
    per_slice = slice_efficiency(entries)
    total_labels = sum(r["labels"] for r in per_slice)
    total_gain = round(sum(r["gain"] for r in per_slice), 4)
    overall_per_k = round(total_gain / total_labels * 1000.0, 4) if total_labels else 0.0
    wasted = [r["slice"] for r in per_slice if r["labels"] > 0 and r["gain"] <= 0]
    return {"per_slice": per_slice, "total_labels": total_labels, "total_gain": total_gain,
            "overall_gain_per_1k": overall_per_k, "n_slices": len(per_slice),
            "best_slice": per_slice[0]["slice"] if per_slice else None,
            "wasted_spend_slices": wasted}
