"""The benchmarks page 500'd the moment it had a real benchmark in it.

  File "services/forgyx/gate.py", line 57, in acc
  TypeError: float() argument must be a string or a real number, not 'NoneType'

`pareto_rank` read accuracy as `float(b.get("map50", 0.0))`, where the default only applies when the key is
absent. benchmark_matrix always sets the key, to None when nothing has scored that model, so the default
never fired.

It survived because every row in the table was seeded demo data carrying an accuracy reference. The first
honestly-recorded benchmark, an ONNX export nobody has scored yet, had `map50: None` and took the endpoint
down. Fabricated data hiding a real defect, which is the same thing the artifact audit was written for.

The fix is not a zero. Zero says the model was measured and is useless; absent says nobody has scored it on
this target. A Pareto plot that conflates them recommends against a model on evidence that does not exist.
"""

from __future__ import annotations

from services.forgyx.gate import pareto_rank


def _b(name: str, p95: float | None, map50: float | None) -> dict:
    return {"model_version": name, "latency_ms": {"p95": p95} if p95 is not None else {}, "map50": map50}


def test_a_benchmark_with_no_accuracy_does_not_crash_the_ranking():
    """The reported 500, exactly."""
    out = pareto_rank([_b("real-export", 77.3, None)])
    assert out[0]["pareto_rank"] is None


def test_an_unmeasured_benchmark_is_marked_rather_than_scored_zero():
    """Zero would rank it dominated by everything, which is a claim about the model rather than about the
    evidence."""
    out = pareto_rank([_b("unmeasured", 10.0, None)])
    assert out[0]["unranked"] is True
    assert "no map50 measured" in out[0]["unranked_reason"]


def test_unmeasured_rows_still_come_back():
    """Dropping them would hide a real artifact from the page whose job is listing artifacts."""
    out = pareto_rank([_b("a", 5.0, 0.8), _b("b", 9.0, None)])
    assert {r["model_version"] for r in out} == {"a", "b"}


def test_unmeasured_rows_sort_last():
    out = pareto_rank([_b("unmeasured", 1.0, None), _b("measured", 50.0, 0.4)])
    assert [r["model_version"] for r in out] == ["measured", "unmeasured"]


def test_an_unmeasured_row_does_not_dominate_a_measured_one():
    # It is fast and unscored. Letting it onto the front would put an unproven model ahead of a proven one.
    out = pareto_rank([_b("fast-unscored", 1.0, None), _b("slow-scored", 90.0, 0.9)])
    scored = next(r for r in out if r["model_version"] == "slow-scored")
    assert scored["pareto_rank"] == 0


def test_the_front_is_still_computed_among_the_measured():
    out = pareto_rank([
        _b("best", 10.0, 0.9),
        _b("slower-worse", 40.0, 0.5),
        _b("unscored", 5.0, None),
    ])
    by = {r["model_version"]: r for r in out}
    assert by["best"]["pareto_rank"] == 0
    assert by["slower-worse"]["pareto_rank"] == 1


def test_a_missing_latency_is_treated_as_infinitely_slow_not_a_crash():
    """A device that reported accuracy and no timing must not take the page down either."""
    out = pareto_rank([_b("no-timing", None, 0.7), _b("timed", 10.0, 0.7)])
    assert [r["model_version"] for r in out] == ["timed", "no-timing"]


def test_a_null_latency_value_is_handled_like_a_missing_one():
    out = pareto_rank([{"model_version": "x", "latency_ms": {"p95": None}, "map50": 0.5}])
    assert out[0]["pareto_rank"] == 0


def test_an_empty_table_is_empty_rather_than_an_error():
    assert pareto_rank([]) == []
