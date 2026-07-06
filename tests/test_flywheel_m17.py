"""Flywheel M17 tests: label-budget apportionment that sums exactly and floors safety-critical demands,
collection tasking that ranks starved ODD cells with target counts, and the adaptive cycle that routes VERDYX
regressions to labeling and SIEVYX gaps to collection (and holds the budget when nothing regressed)."""

from services.flywheel.allocator import allocate_label_budget
from services.flywheel.controller import adaptive_cycle
from services.flywheel.tasking import collection_tasks


def test_allocation_sums_exactly_and_floors_safety():
    demands = [
        {"slice": "car_day", "weight": 0.6, "safety_weight": 1.0},
        {"slice": "vru_night", "weight": 0.1, "safety_weight": 2.0},   # small weight but safety-critical
        {"slice": "sign_rain", "weight": 0.3, "safety_weight": 1.0},
    ]
    alloc = allocate_label_budget(demands, total_budget=100, safety_floor=20)
    assert sum(a["labels"] for a in alloc) == 100          # exact apportionment
    vru = next(a for a in alloc if a["slice"] == "vru_night")
    assert vru["labels"] >= 20                              # safety floor honored despite low weight
    assert allocate_label_budget([], 100) == []


def test_collection_tasking_ranks_and_targets():
    gaps = [
        {"cell": "day_clear_highway", "gap": 0.05, "missing": False},
        {"cell": "night_rain_junction", "gap": 0.04, "missing": True},   # smaller gap, but safety keywords
    ]
    tasks = collection_tasks(gaps, total_samples=10000)
    # the safety-weighted night-rain-junction cell outranks the larger benign gap
    assert tasks[0]["cell"] == "night_rain_junction"
    assert tasks[0]["missing"] is True
    # target count closes the gap against the current fleet size
    assert tasks[0]["target_count"] == 400                 # ceil(0.04 * 10000)


def test_adaptive_cycle_routes_and_holds():
    regressions = [
        {"slice": "vru_night", "delta": -0.10, "protected": True},
        {"slice": "car_day", "delta": -0.03, "protected": False},
    ]
    gaps = [{"cell": "night_rain_junction", "gap": 0.04, "missing": True}]
    plan = adaptive_cycle(regressions, gaps, total_label_budget=200, total_fleet_samples=10000,
                          safety_slices=["vru_night"], safety_floor=50)
    assert plan["allocated"] == 200                         # a labeling demand exists, budget is spent
    vru = next(a for a in plan["allocation"] if a["slice"] == "vru_night")
    assert vru["labels"] >= 50                              # protected slice floored
    assert plan["collection_tasks"][0]["cell"] == "night_rain_junction"
    # no regressions -> budget is held, not spent blindly, and the cycle says so
    held = adaptive_cycle([], gaps, total_label_budget=200, total_fleet_samples=10000)
    assert held["allocated"] == 0 and held["held"] == 200
    assert "held" in held["rationale"]
