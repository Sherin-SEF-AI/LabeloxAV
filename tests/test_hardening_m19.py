"""Hardening M19 tests: per-plane SLO evaluation (breach + unobserved-is-breach), byte-stable reproducibility
with float-jitter tolerance and divergence naming, the label-budget efficiency report (ROI ranking + wasted
spend), and an end-to-end scale run of the flywheel over a large synthetic fleet batch that must conserve the
label budget exactly."""

from services.flywheel.controller import adaptive_cycle
from services.hardening.efficiency import efficiency_report
from services.hardening.repro import check_reproducible, content_hash
from services.hardening.slo import evaluate_slos


def test_plane_slo_evaluation():
    ok = evaluate_slos("verdyx", {"eval_p95_s": 400.0, "safety_recall": 0.62})
    assert ok["met"] is True and ok["breaches"] == []
    # safety recall under the floor breaches
    bad = evaluate_slos("verdyx", {"eval_p95_s": 400.0, "safety_recall": 0.40})
    assert bad["met"] is False and bad["breaches"][0]["metric"] == "safety_recall"
    # an unobserved metric is a breach, not assumed healthy
    missing = evaluate_slos("verdyx", {"eval_p95_s": 400.0})
    assert missing["met"] is False and any(b["reason"] == "unobserved" for b in missing["breaches"])


def test_byte_stable_reproducibility():
    build_a = {"commit": "c1", "samples": 1000, "metrics": {"map50": 0.612345678}}
    # a re-build with sub-precision float jitter must still be reproducible
    build_b = {"commit": "c1", "samples": 1000, "metrics": {"map50": 0.6123456781}}
    r = check_reproducible(build_a, build_b)
    assert r["reproducible"] is True and content_hash(build_a) == content_hash(build_b)
    # a real difference is caught and the diverging field named
    build_c = {"commit": "c1", "samples": 1001, "metrics": {"map50": 0.612345678}}
    d = check_reproducible(build_a, build_c)
    assert d["reproducible"] is False and d["first_divergence"] == "samples"


def test_label_efficiency_report():
    entries = [
        {"slice": "vru_night", "labels": 500, "map_before": 0.40, "map_after": 0.55},   # strong ROI
        {"slice": "car_day", "labels": 2000, "map_before": 0.80, "map_after": 0.805},   # weak ROI
        {"slice": "sign_rain", "labels": 300, "map_before": 0.60, "map_after": 0.58},   # negative ROI
    ]
    rep = efficiency_report(entries)
    assert rep["best_slice"] == "vru_night"
    assert "sign_rain" in rep["wasted_spend_slices"]
    vru = next(r for r in rep["per_slice"] if r["slice"] == "vru_night")
    assert vru["gain_per_1k"] == 0.3                      # 0.15 mAP gain over 500 labels -> 0.30 per 1k labels
    assert rep["total_labels"] == 2800


def test_flywheel_scale_conserves_budget():
    # e2e scale: 400 regressed slices and 400 ODD gaps through one adaptive cycle
    regressions = [{"slice": f"s{i}", "delta": -0.01 - (i % 7) * 0.005,
                    "protected": i % 50 == 0} for i in range(400)]
    gaps = [{"cell": f"night_rain_cell_{i}" if i % 3 == 0 else f"cell_{i}",
             "gap": 0.001 + (i % 11) * 0.0003, "missing": i % 9 == 0} for i in range(400)]
    budget = 100000
    plan = adaptive_cycle(regressions, gaps, total_label_budget=budget, total_fleet_samples=1_000_000,
                          safety_slices=[f"s{i}" for i in range(0, 400, 50)], safety_floor=100)
    # the whole budget is apportioned exactly, at scale, with no leak or overspend
    assert sum(a["labels"] for a in plan["allocation"]) == budget
    assert plan["allocated"] == budget and plan["held"] == 0
    assert len(plan["collection_tasks"]) == 400
    # protected slices are floored even in a crowded field
    protected = [a for a in plan["allocation"] if a["slice"] in {f"s{i}" for i in range(0, 400, 50)}]
    assert all(a["labels"] >= 100 for a in protected)
