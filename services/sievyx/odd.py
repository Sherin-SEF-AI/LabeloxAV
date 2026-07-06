"""SIEVYX ODD coverage-gap detection (M12). A dataset is only as safe as its coverage of the operational
design domain. This compares the fleet's actual scenario-cell coverage (illumination x weather x road-type x
density x region cells, from SIEVYX ScenarioTags) against a target ODD distribution and reports the
underrepresented cells, so collection and mining can be tasked exactly where the dataset is thin. Pure over two
distributions, so it is testable."""

from __future__ import annotations


def coverage_gaps(fleet_counts: dict[str, int], target_odd: dict[str, float], min_gap: float = 0.02) -> dict:
    """Compare fleet coverage to the target ODD. fleet_counts maps a scenario cell to its sample count;
    target_odd maps a cell to its target fraction (need not sum to 1). Returns the underrepresented cells,
    largest gap first, plus the cells entirely missing from the fleet."""
    total = sum(fleet_counts.values())
    tsum = sum(target_odd.values()) or 1.0
    gaps = []
    for cell, target in target_odd.items():
        target_frac = target / tsum
        actual_frac = (fleet_counts.get(cell, 0) / total) if total else 0.0
        gap = target_frac - actual_frac
        if gap > min_gap:
            gaps.append({"cell": cell, "target_frac": round(target_frac, 4),
                         "actual_frac": round(actual_frac, 4), "gap": round(gap, 4),
                         "count": fleet_counts.get(cell, 0),
                         "missing": fleet_counts.get(cell, 0) == 0})
    gaps.sort(key=lambda g: g["gap"], reverse=True)
    return {"total_samples": total, "n_cells": len(target_odd), "n_gaps": len(gaps),
            "coverage_score": round(1.0 - sum(g["gap"] for g in gaps), 4), "gaps": gaps}
