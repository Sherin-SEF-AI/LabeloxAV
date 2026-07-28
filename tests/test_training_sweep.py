"""Hyperparameter search, and the two training-run behaviours that lost work.

Training exposed three knobs, each set once. There was no search of any kind, so tuning meant running a job
by hand, noting the number, editing the request, and running it again, with no record of what had been tried.

Two related losses: a crashed run was flipped back to pending and restarted at epoch zero, discarding every
epoch already paid for even though ultralytics had written a checkpoint each epoch; and only the latest
metric point was kept, so the shape of a run (where it plateaued, when it collapsed) was unreadable
afterwards.

Pure unit tests over the planning and ranking logic; no GPU and no database."""
from __future__ import annotations

import pytest

from services.training.sweep import (
    MAX_TRIALS,
    expand_grid,
    plan_trials,
    rank_trials,
    sample_random,
    summarize_sweep,
)


def test_grid_is_the_full_cross_product():
    trials = expand_grid({"epochs": [10, 20], "imgsz": [640, 960]})
    assert len(trials) == 4
    assert {"epochs": 10, "imgsz": 640} in trials
    assert {"epochs": 20, "imgsz": 960} in trials


def test_grid_order_is_deterministic():
    # Two runs of the same space must number their trials identically, or trial 3 means nothing.
    space = {"b": [1, 2], "a": [3, 4]}
    assert expand_grid(space) == expand_grid(space)


def test_empty_space_is_a_single_default_trial():
    # Sweeping nothing should run the base configuration once, not zero times.
    assert expand_grid({}) == [{}]
    assert plan_trials({})["planned"] == 1


def test_random_search_samples_without_replacement():
    space = {"epochs": [5, 10, 20, 40], "imgsz": [320, 640, 960]}
    trials = sample_random(space, 6, seed=1)
    assert len(trials) == 6
    # paying twice for an identical trial is pure waste on a serialized GPU
    assert len({tuple(sorted(t.items())) for t in trials}) == 6


def test_random_search_returns_the_whole_space_when_it_is_smaller_than_the_budget():
    space = {"epochs": [5, 10]}
    assert len(sample_random(space, 10)) == 2


def test_random_search_is_reproducible_for_a_seed():
    space = {"epochs": [5, 10, 20, 40], "imgsz": [320, 640, 960]}
    assert sample_random(space, 4, seed=42) == sample_random(space, 4, seed=42)


def test_an_oversized_space_is_capped_and_says_so():
    # Crossing four lists can enqueue hundreds of GPU-hours. Truncating silently would look like the sweep
    # simply finished.
    plan = plan_trials({"a": list(range(100))}, max_trials=5)
    assert plan["planned"] == 5
    assert plan["requested"] == 100
    assert plan["truncated"] is True


def test_a_space_within_the_cap_is_not_marked_truncated():
    plan = plan_trials({"epochs": [10, 20]})
    assert plan["truncated"] is False and plan["planned"] == 2


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown sweep method"):
        plan_trials({"epochs": [1]}, method="bayesian")


def test_default_cap_is_bounded():
    assert 1 <= MAX_TRIALS <= 256


# ---------------- ranking ----------------

def _trial(job_id: str, map50=None):
    return {"job_id": job_id, "hparams": {"epochs": 10},
            "metrics": ({"map50": map50} if map50 is not None else {})}


def test_ranking_is_best_first():
    ranked = rank_trials([_trial("a", 0.3), _trial("b", 0.5), _trial("c", 0.4)])
    assert [r["job_id"] for r in ranked] == ["b", "c", "a"]


def test_a_trial_with_no_metric_is_excluded_rather_than_scored_zero():
    # A crashed or cancelled trial has no metric. Scoring it zero would make it look merely bad rather than
    # absent, and it would sit at the bottom of a ranking as though it had been evaluated.
    s = summarize_sweep([_trial("a", 0.3), _trial("crashed")])
    assert s["n_scored"] == 1 and s["n_unscored"] == 1
    assert [r["job_id"] for r in s["ranking"]] == ["a"]


def test_summary_of_an_all_failed_sweep_has_no_winner():
    s = summarize_sweep([_trial("a"), _trial("b")])
    assert s["best"] is None and s["n_scored"] == 0


def test_ranking_can_use_a_different_metric():
    trials = [{"job_id": "a", "hparams": {}, "metrics": {"map50_mask": 0.2}},
              {"job_id": "b", "hparams": {}, "metrics": {"map50_mask": 0.6}}]
    assert summarize_sweep(trials, metric="map50_mask")["best"]["job_id"] == "b"


# ---------------- run-loss behaviours ----------------

def test_resume_is_on_by_default():
    # An orphaned run has a checkpoint on disk from every completed epoch; restarting at zero throws that
    # GPU time away.
    from core.config import get_settings

    assert get_settings().training.resume_orphaned_runs is True


def test_metric_curve_is_appended_not_replaced():
    # Reproduces the persistence rule in services/training/jobs.py: the curve accumulates so the shape of a
    # run survives, bounded so a long sweep cannot grow a row without limit.
    metrics: dict = {}
    for epoch in range(1, 6):
        curve = list(metrics.get("curve") or [])
        curve.append({"epoch": epoch, "map50": epoch / 10})
        metrics["curve"] = curve[-500:]

    assert len(metrics["curve"]) == 5
    assert metrics["curve"][0]["epoch"] == 1 and metrics["curve"][-1]["epoch"] == 5


def test_metric_curve_is_bounded_and_keeps_the_recent_shape():
    curve: list[dict] = []
    for epoch in range(1, 601):
        curve.append({"epoch": epoch})
        curve = curve[-500:]
    assert len(curve) == 500
    assert curve[-1]["epoch"] == 600     # dropping from the front keeps the recent end
