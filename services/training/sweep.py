"""Hyperparameter sweeps: enqueue a set of training jobs over a parameter space and pick the winner.

Training exposed exactly three knobs (epochs, imgsz, batch), each set once, so tuning meant running a job by
hand, writing the number down, editing the request, and running it again. There was no search of any kind:
no grid, no random, no record of what was tried.

A sweep is deliberately built from ordinary training jobs rather than a parallel execution path. The worker
holds a Postgres advisory lock as a global GPU mutex, so trials serialize on the one GPU anyway, and reusing
the job queue means every trial gets the existing gating, cancellation, progress, and registry behaviour for
free. The sweep is bookkeeping over jobs, which is all it should be.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from core.logging import get_logger

log = get_logger("sweep")

# A sweep that enqueues hundreds of GPU-hours because someone crossed four lists is a foot-gun, so the
# expansion is capped and the caller is told rather than silently truncated.
MAX_TRIALS = 64


def expand_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination in the space, in a deterministic order.

    Sorted by key so two runs of the same space produce trials in the same order, which is what makes a
    sweep re-runnable and its trial numbering meaningful.
    """
    if not space:
        return [{}]
    keys = sorted(space)
    combos = itertools.product(*(space[k] for k in keys))
    return [dict(zip(keys, values, strict=True)) for values in combos]


def sample_random(space: dict[str, list[Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    """n distinct points sampled from the space.

    Random search beats grid search when only a few dimensions matter, because a grid spends its budget
    re-testing the same value of an unimportant parameter. Sampling without replacement avoids paying twice
    for an identical trial; if the space is smaller than n, the whole space is returned rather than looping.
    """
    if not space:
        return [{}]
    rng = random.Random(seed)
    grid = expand_grid(space)
    if len(grid) <= n:
        return grid
    return rng.sample(grid, n)


def plan_trials(space: dict[str, list[Any]], *, method: str = "grid", max_trials: int = MAX_TRIALS,
                seed: int = 7) -> dict:
    """Turn a parameter space into a concrete, bounded trial list."""
    if method not in ("grid", "random"):
        raise ValueError(f"unknown sweep method {method!r}; use 'grid' or 'random'")
    full = expand_grid(space)
    if method == "random":
        trials = sample_random(space, max_trials, seed)
    else:
        trials = full[:max_trials]
    return {
        "method": method,
        "trials": trials,
        "requested": len(full),
        "planned": len(trials),
        # Say so when the space was larger than the cap, rather than quietly running a prefix of it.
        "truncated": len(trials) < len(full),
        "max_trials": max_trials,
    }


async def start_sweep(*, name: str, space: dict[str, list[Any]], base: dict,
                      method: str = "grid", max_trials: int = MAX_TRIALS, seed: int = 7) -> dict:
    """Enqueue one training job per trial. `base` is the shared TrainJobSpec payload.

    Each trial's hparams are the base hparams overlaid with the trial point, and each gets its own run name
    so the trials do not overwrite each other's weights, which is what would happen if they shared one.
    """
    from services.training.jobs import TrainJobSpec, enqueue_job

    plan = plan_trials(space, method=method, max_trials=max_trials, seed=seed)
    if plan["truncated"]:
        log.warning("sweep.truncated", requested=plan["requested"], planned=plan["planned"],
                    max_trials=max_trials)

    job_ids: list[str] = []
    for i, point in enumerate(plan["trials"]):
        hparams = {**(base.get("hparams") or {}), **point}
        spec = TrainJobSpec(
            purpose=base.get("purpose", "perception"),
            task_type=base.get("task_type", "detection"),
            compute_target=base.get("compute_target", "local"),
            dataset_spec=dict(base.get("dataset_spec") or {}),
            base_weights=base.get("base_weights"),
            hparams=hparams,
            gate=dict(base.get("gate") or {}),
            # A sweep trial must never auto-promote: the point is to compare trials first, and promoting the
            # first one to finish would make the winner a function of scheduling order.
            promote=False,
            notes=f"sweep '{name}' trial {i + 1}/{plan['planned']} {point}",
        )
        job_ids.append(str(await enqueue_job(spec)))

    log.info("sweep.started", name=name, trials=len(job_ids), method=method)
    return {"sweep": name, "method": method, "trials": len(job_ids), "job_ids": job_ids,
            "points": plan["trials"], "truncated": plan["truncated"], "requested": plan["requested"]}


def rank_trials(results: list[dict], metric: str = "map50") -> list[dict]:
    """Order finished trials best-first on a metric, ignoring any that never produced one.

    A crashed or cancelled trial has no metric. Treating it as zero would let it look merely bad rather than
    absent, so it is dropped from the ranking and reported separately by summarize_sweep.
    """
    scored = [r for r in results if isinstance((r.get("metrics") or {}).get(metric), int | float)]
    return sorted(scored, key=lambda r: float(r["metrics"][metric]), reverse=True)


def summarize_sweep(results: list[dict], metric: str = "map50") -> dict:
    """The winner, the ranking, and an honest count of what did not finish."""
    ranked = rank_trials(results, metric)
    unscored = [r for r in results if r not in ranked]
    return {
        "metric": metric,
        "n_trials": len(results),
        "n_scored": len(ranked),
        "n_unscored": len(unscored),
        "best": ranked[0] if ranked else None,
        "ranking": [{"job_id": r.get("job_id"), "hparams": r.get("hparams"),
                     metric: r["metrics"][metric]} for r in ranked],
    }
