"""Flywheel live-signal tests: the pure functions that turn real corpus counts into controller signals. Class
starvation produces labeling demands only for safety-critical classes below their floor (protected for
VRU/animal), and scene aggregation keys ODD cells by time-of-day x weather, skipping untagged frames."""

from services.flywheel.signals import scene_fleet_counts, starvation_demands


def test_starvation_demands_flags_only_starved_safety_classes():
    total = 100_000
    counts = [
        {"name": "cattle", "l1": "animal", "class_id": 31, "count": 32},        # 0.03% -> starved, protected
        {"name": "pedestrian", "l1": "vru", "class_id": 25, "count": 5000},     # 5% -> fine
        {"name": "motorcycle", "l1": "two_wheeler", "class_id": 1, "count": 40},# 0.04% -> starved, not protected
        {"name": "sedan", "l1": "car", "class_id": 40, "count": 3},             # not safety-critical -> ignored
    ]
    demands = starvation_demands(counts, total, min_share=0.003)
    slices = {d["slice"] for d in demands}
    assert "cattle" in slices and "motorcycle" in slices     # both below the 0.3% floor
    assert "pedestrian" not in slices                         # above the floor
    assert "sedan" not in slices                              # not a safety class, never a demand
    cattle = next(d for d in demands if d["slice"] == "cattle")
    assert cattle["protected"] is True                        # animal is protected
    moto = next(d for d in demands if d["slice"] == "motorcycle")
    assert moto["protected"] is False                         # two_wheeler is safety-relevant but not floored
    # worst-starved first (more negative delta)
    assert demands[0]["delta"] <= demands[-1]["delta"]


def test_scene_fleet_counts_keys_and_skips_untagged():
    rows = [
        {"time_of_day": "night", "weather": "rain"},
        {"time_of_day": "night", "weather": "rain"},
        {"time_of_day": "day", "weather": "clear"},
        {"time_of_day": "night", "weather": None},     # half-tagged -> skipped
        {"time_of_day": None, "weather": None},         # untagged -> skipped
    ]
    counts = scene_fleet_counts(rows)
    assert counts == {"night_rain": 2, "day_clear": 1}
