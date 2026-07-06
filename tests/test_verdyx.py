"""M7 acceptance: a challenger that improves aggregate mAP but regresses the pedestrian-at-night protected
slice is rejected by the VERDYX gate; a clean improvement promotes; a flat one needs review."""

from services.verdyx.verdict import slice_matrix, slice_verdict

PROTECTED = ["pedestrian_night", "autorickshaw_glare"]


def _eval(agg, slices):
    return {"aggregate": {"map50": agg}, "per_slice": {k: {"map": v} for k, v in slices.items()}}


def test_reject_on_protected_slice_regression_despite_aggregate_gain():
    champion = _eval(0.50, {"pedestrian_night": 0.60, "autorickshaw_glare": 0.55, "sedan_day": 0.70})
    challenger = _eval(0.56, {"pedestrian_night": 0.40, "autorickshaw_glare": 0.56, "sedan_day": 0.80})
    v = slice_verdict(champion, challenger, PROTECTED)
    assert v["verdict"] == "reject"
    assert v["regressed_slices"][0]["slice"] == "pedestrian_night"
    assert v["beats_aggregate"] is True  # aggregate improved, but the gate still rejects


def test_promote_on_clean_improvement():
    champion = _eval(0.50, {"pedestrian_night": 0.60, "autorickshaw_glare": 0.55})
    challenger = _eval(0.55, {"pedestrian_night": 0.62, "autorickshaw_glare": 0.57})
    v = slice_verdict(champion, challenger, PROTECTED)
    assert v["verdict"] == "promote"


def test_needs_review_when_no_uplift_and_no_regression():
    champion = _eval(0.50, {"pedestrian_night": 0.60})
    challenger = _eval(0.501, {"pedestrian_night": 0.60})  # below min uplift
    v = slice_verdict(champion, challenger, PROTECTED)
    assert v["verdict"] == "needs_review"


def test_small_slice_drop_within_tolerance_still_promotes():
    champion = _eval(0.50, {"pedestrian_night": 0.60})
    challenger = _eval(0.55, {"pedestrian_night": 0.59})  # 0.01 drop, within 0.02 tol
    v = slice_verdict(champion, challenger, PROTECTED)
    assert v["verdict"] == "promote"


def test_slice_matrix_states():
    champion = _eval(0.5, {"a": 0.6, "b": 0.5})
    challenger = _eval(0.5, {"a": 0.4, "b": 0.7})
    rows = {r["slice"]: r for r in slice_matrix(champion, challenger, ["a", "b"])}
    assert rows["a"]["state"] == "worse" and rows["b"]["state"] == "better"
