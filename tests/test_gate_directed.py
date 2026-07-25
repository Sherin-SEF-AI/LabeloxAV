"""Gate-directed labeling: a blocked promotion has to name the classes it blocked on, and the batch has to be
built from that diagnosis rather than from corpus share.

The regression these guard is the one that cost five operational iterations: the flywheel allocated label
budget by how small a class's share of the corpus was, so pedestrian at 514 instances and 0.02 recall read as
healthy while the safety gate blocked on it. The demand must follow the gate's arithmetic, not the histogram.
"""

from types import SimpleNamespace

import pytest

from services.autolabel.ontology import get_ontology
from services.flywheel.gate_signals import recall_demands

RCFG = SimpleNamespace(safety_recall_floor=0.5, safety_recall_max_drop=0.05,
                       safety_class_drop={"vru": 0.1, "animal": 0.1, "_default": 0.15},
                       require_safety_recall=True)


@pytest.fixture(scope="module")
def onto():
    return get_ontology()


def test_class_below_the_floor_becomes_a_demand(onto):
    chal = {"per_class_recall": {"cattle": 0.14, "pedestrian": 0.73, "sedan": 0.0}}
    demands = recall_demands(chal, None, onto, RCFG)

    assert [d["class_name"] for d in demands] == ["cattle"]
    d = demands[0]
    assert d["kind"] == "floor_miss"
    assert d["observed"] == 0.14 and d["target"] == 0.5
    assert d["deficit"] == pytest.approx(0.36)


def test_a_non_safety_class_never_becomes_a_demand(onto):
    """sedan sits at 0.0 recall, far below the floor, but it is not VRU or animal so the safety gate does not
    block on it and a reviewer must not be sent to label it in a safety batch."""
    demands = recall_demands({"per_class_recall": {"sedan": 0.0, "truck": 0.0}}, None, onto, RCFG)
    assert demands == []


def test_share_healthy_but_recall_starved_class_is_still_demanded(onto):
    """The iteration 1-5 failure in one assertion: pedestrian was well represented in the corpus and would
    look healthy to a share-based signal, yet its recall is what blocked promotion."""
    demands = recall_demands({"per_class_recall": {"pedestrian": 0.02}}, None, onto, RCFG)
    assert [d["class_name"] for d in demands] == ["pedestrian"]
    assert demands[0]["kind"] == "floor_miss"


def test_regression_above_the_floor_is_caught(onto):
    """Above the floor a class can still block by losing ground against the incumbent."""
    chal = {"per_class_recall": {"rider": 0.60}}
    champ = {"per_class_recall": {"rider": 0.80}}
    demands = recall_demands(chal, champ, onto, RCFG)

    assert [d["class_name"] for d in demands] == ["rider"]
    d = demands[0]
    assert d["kind"] == "regression"
    assert d["baseline"] == 0.8
    assert d["target"] == pytest.approx(0.75)  # baseline - max_drop


def test_a_drop_inside_tolerance_is_not_a_demand(onto):
    demands = recall_demands({"per_class_recall": {"rider": 0.78}},
                             {"per_class_recall": {"rider": 0.80}}, onto, RCFG)
    assert demands == []


def test_missing_per_class_recall_manufactures_no_work(onto):
    """Fail-closed like the gate: without per-class recall we cannot say which class is short, and guessing
    would send a human to label the wrong thing."""
    assert recall_demands({}, None, onto, RCFG) == []
    assert recall_demands({"map50": 0.3}, None, onto, RCFG) == []


def test_demands_rank_by_deficit_and_floor_misses_outweigh_regressions(onto):
    chal = {"per_class_recall": {"cattle": 0.0, "pedestrian": 0.40, "rider": 0.60}}
    champ = {"per_class_recall": {"rider": 0.80}}
    demands = recall_demands(chal, champ, onto, RCFG)

    # cattle (deficit 0.5, floor) > pedestrian (deficit 0.1, floor) > rider (deficit 0.15, regression):
    # rider's raw deficit beats pedestrian's, but a floor miss is weighted the more urgent failure.
    assert [d["class_name"] for d in demands] == ["cattle", "pedestrian", "rider"]
    assert all(d["safety_weight"] == 2.0 for d in demands)


def test_demand_shape_is_what_the_allocator_consumes(onto):
    """The demand plugs into services/flywheel/allocator.py unchanged, so budgeting stays single-sourced."""
    from services.flywheel.allocator import allocate_label_budget

    demands = recall_demands({"per_class_recall": {"cattle": 0.0, "rider": 0.3}}, None, onto, RCFG)
    alloc = allocate_label_budget(demands, 500, safety_floor=50)

    assert sum(a["labels"] for a in alloc) == 500
    assert {a["slice"] for a in alloc} == {"cattle", "rider"}
    assert all(a["labels"] >= 50 for a in alloc)  # every blocked class is safety-critical, so all are floored
