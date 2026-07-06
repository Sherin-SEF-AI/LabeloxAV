"""Adaptive flywheel controller (M17). This closes the data-engine loop: it consumes VERDYX safety regressions
and SIEVYX ODD coverage gaps, decides where the finite label budget and collection effort should go this cycle,
and emits an auditable plan. Labeling fixes slices where the data exists but the model is wrong; collection
fixes cells where the data does not exist. A regression on a protected slice is weighted so it is never
starved. Pure over the two signal sets, so the whole cycle is deterministic and testable; persistence to the
FlywheelCycle ledger lives in run.py."""

from __future__ import annotations

from services.flywheel.allocator import allocate_label_budget
from services.flywheel.tasking import collection_tasks


def _demand_from_regression(reg: dict, safety_slices: set[str]) -> dict:
    """A VERDYX regression becomes a label demand: the worse the drop, the larger the weight; a protected slice
    carries a safety_weight so the allocator floors it."""
    slice_id = reg.get("slice", "")
    drop = abs(float(reg.get("delta", 0.0)))
    protected = reg.get("protected", False) or slice_id in safety_slices
    return {"slice": slice_id, "weight": max(drop, 1e-3), "safety_weight": 2.0 if protected else 1.0,
            "reason": f"VERDYX regression {drop:.3f}" + (" (protected)" if protected else "")}


def adaptive_cycle(regressions: list[dict], odd_gaps: list[dict], total_label_budget: int,
                   total_fleet_samples: int, safety_slices: list[str] | None = None,
                   safety_floor: int = 0) -> dict:
    """Run one adaptive cycle.

    regressions is the VERDYX shadow-triage / safety list ([{slice, delta, protected}]); odd_gaps is the SIEVYX
    coverage-gap list. Returns the label-budget allocation across regressed slices, the ranked collection tasks
    for the ODD gaps, and a rationale. When there are no regressions the whole label budget is unallocated
    (nothing to fix by labeling), which is itself a signal the cycle reports rather than spending blindly."""
    safety = set(safety_slices or [])
    demands = [_demand_from_regression(r, safety) for r in regressions]
    allocation = allocate_label_budget(demands, total_label_budget, safety_floor) if demands else []
    tasks = collection_tasks(odd_gaps, total_fleet_samples)

    allocated = sum(a["labels"] for a in allocation)
    n_protected = sum(1 for r in regressions if r.get("protected") or r.get("slice") in safety)
    rationale = (
        f"{len(regressions)} regressed slices ({n_protected} protected) drew {allocated}/{total_label_budget} "
        f"labels; {len(tasks)} ODD gaps queued for collection."
        if demands else
        f"no regressions this cycle; {total_label_budget} label budget held; {len(tasks)} ODD gaps queued for "
        f"collection."
    )
    return {"label_budget": total_label_budget, "allocation": allocation, "collection_tasks": tasks,
            "allocated": allocated, "held": total_label_budget - allocated, "rationale": rationale}
