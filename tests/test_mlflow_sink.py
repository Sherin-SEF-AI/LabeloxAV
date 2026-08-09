"""Publishing model numbers, and the fields that stop one meaning the wrong thing.

Every number this system produces lives only in Postgres, so a promotion cannot be held against last month's
and an external model has no shared home for its metrics. What is published is chosen by what actually went
wrong here rather than by what is easy to log.

The gold id and the scored population travel with the number because this corpus has already had a sealed set
shrink from 400 objects to 47 while its mAP kept reading confidently, and because a model whose vocabulary
covers half the classes is scored over half the set: DashLab's detector reports 152 of 302.

Harness divergence is a metric rather than a note because it is the signal that a measurement is wrong rather
than a model, and it blocked every promotion here for months. A history that recorded only mAP would show a
tidy run of numbers that did not mean anything.

Nothing here may fail a promotion, so every path is exercised for "the tracking server is down".
"""

from __future__ import annotations

import pytest

from services.integrations import mlflow_sink
from services.integrations.mlflow_sink import _clean, log_evaluation, log_promotion, status

METRICS = {
    "map50": 0.3471, "map50_95": 0.299, "precision": 0.326, "recall": 0.545,
    "safe_miou": 0.55, "prediction_plane_ap50": 0.221, "harness_delta": 0.1261,
    "harness_divergent": True,
    "gold_declared": 302, "gold_resolvable": 302, "prediction_plane_scored": 152,
    "per_class_recall": {"truck": 0.5, "autorickshaw": 0.0},
    "per_class_ap50": {"truck": 0.6},
}


class _Run:
    def __init__(self): self.info = type("I", (), {"run_id": "run-1"})()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Mlflow:
    """Records what would have been sent, so the assertions are about content rather than a live server."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.tags: dict = {}
        self.params: dict = {}
        self.metrics: dict = {}

    def start_run(self, run_name=None):
        if self.fail:
            raise RuntimeError("tracking server unreachable")
        self.run_name = run_name
        return _Run()

    def set_tags(self, d): self.tags.update(d)
    def log_params(self, d): self.params.update(d)
    def log_metrics(self, d): self.metrics.update(d)
    def log_metric(self, k, v): self.metrics[k] = v


@pytest.fixture
def sink(monkeypatch):
    fake = _Mlflow()
    monkeypatch.setitem(mlflow_sink._state, "enabled", True)
    monkeypatch.setitem(mlflow_sink._state, "checked", True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    return fake


def test_an_evaluation_carries_the_gold_set_it_was_measured_over(sink):
    """A number without its population is the thing this exists to stop shipping."""
    log_evaluation(model_version="m1", gold_id="gold-abc", metrics=METRICS)
    assert sink.params["gold_id"] == "gold-abc"
    assert sink.params["gold_declared"] == 302
    assert sink.params["gold_scored"] == 152, "how much of the set this model could actually be scored on"


def test_harness_divergence_is_a_metric_not_a_note(sink):
    """It has to be filterable and plottable over time: it is what says the measurement is wrong."""
    log_evaluation(model_version="m1", gold_id="g", metrics=METRICS)
    assert sink.metrics["harness_divergent"] == 1.0
    assert sink.metrics["harness_delta"] == pytest.approx(0.1261)


def test_per_class_numbers_are_published_individually(sink):
    """A single mAP hides that autorickshaw recall is zero, which is the finding worth keeping."""
    log_evaluation(model_version="m1", gold_id="g", metrics=METRICS)
    assert sink.metrics["recall/autorickshaw"] == 0.0
    assert sink.metrics["recall/truck"] == 0.5
    assert sink.metrics["ap50/truck"] == 0.6


def test_an_external_model_is_tagged_as_one(sink):
    """A model this engine trained and one scored here after being trained elsewhere are different claims."""
    log_evaluation(model_version="dashlab", gold_id="g", metrics=METRICS, tags={"source": "external"})
    assert sink.tags["source"] == "external"


def test_a_refused_promotion_is_published_with_its_reasons(sink):
    """The refusals are the more useful half: every model here was blocked for months, and a history of only
    successful promotions would show nothing at all for that period."""
    gate = {"promote": False, "map_delta": -0.0266, "beats_map": False, "safety_ok": True,
            "recall_ok": False, "reasons": ["safety-class recall below floor 0.5: ['cattle']"]}
    log_promotion(model_version="m1", promoted=False, gate=gate)
    assert sink.tags["promoted"] == "False"
    assert "cattle" in sink.tags["reasons"]
    assert sink.metrics["promoted"] == 0.0
    assert sink.metrics["map_delta"] == pytest.approx(-0.0266)


def test_a_tracking_server_that_is_down_never_fails_the_caller(monkeypatch):
    """An evaluation that ran and a tracking server that did not is a reporting failure. Turning it into a
    promotion failure would put the gate at the mercy of a metrics service."""
    monkeypatch.setitem(mlflow_sink._state, "enabled", True)
    monkeypatch.setitem(mlflow_sink._state, "checked", True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", _Mlflow(fail=True))

    assert log_evaluation(model_version="m1", gold_id="g", metrics=METRICS) is None
    assert log_promotion(model_version="m1", promoted=True, gate={}) is None


def test_it_is_inert_without_a_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setitem(mlflow_sink._state, "enabled", False)
    monkeypatch.setitem(mlflow_sink._state, "checked", False)
    assert log_evaluation(model_version="m1", gold_id="g", metrics=METRICS) is None
    assert status()["enabled"] is False


# ------------------------------------------------------------------------------- value cleaning

def test_non_finite_values_are_dropped():
    """A NaN reaches the server as a string and then sorts and plots as though it were a number, which is
    worse than the metric simply being absent."""
    out = _clean({"a": float("nan"), "b": float("inf"), "c": 0.5})
    assert out == {"c": 0.5}


def test_booleans_become_numbers_so_they_can_be_plotted():
    assert _clean({"divergent": True, "ok": False}) == {"divergent": 1.0, "ok": 0.0}


def test_non_scalars_are_dropped_rather_than_stringified():
    """A dict logged as a metric name is noise that looks like data."""
    assert _clean({"per_class": {"a": 1}, "name": "m1", "n": 3}) == {"n": 3.0}


def test_none_is_dropped():
    assert _clean({"a": None, "b": 1}) == {"b": 1.0}
