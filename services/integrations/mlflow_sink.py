"""Publishing model numbers somewhere they can be compared.

Thirteen registered models and every evaluation this system has produced live only in Postgres, reachable
only through this application's own API. So a promotion cannot be held against last month's, an external
model (DashLab's detector, registered here after being trained elsewhere) has no shared home for its metrics,
and the signed quality certificate rests on a history only this database remembers.

What gets logged is chosen by what went wrong this week rather than by what is easy to log.

**The gold id is a parameter, not a note.** A number is meaningless without the population it was measured
over, and this corpus has already had a gold set shrink from 400 objects to 47 while its mAP kept reading
confidently. `gold_declared`, `gold_resolvable` and `gold_scored` go alongside it for the same reason.

**Harness divergence is a metric.** Two independent harnesses disagreeing about one model on one gold set is
the signal that the measurement is wrong rather than the model, and it is what blocked every promotion here
until it was fixed. A tracking server that recorded only mAP would have shown a tidy history of numbers that
did not mean anything.

Inert without `MLFLOW_TRACKING_URI`. Nothing here may fail a promotion: an evaluation that succeeded and a
tracking server that is down is a reporting problem, not a model problem, so every failure is captured and
swallowed.
"""

from __future__ import annotations

import os
from typing import Any

from core.logging import get_logger
from core.observability import capture

log = get_logger("mlflow_sink")

EXPERIMENT = "labeloxav-models"

_state: dict[str, Any] = {"enabled": False, "checked": False}


def _uri() -> str | None:
    return (os.environ.get("MLFLOW_TRACKING_URI") or "").strip() or None


def enabled() -> bool:
    """Whether there is somewhere to publish to. Resolved once; a tracking server that appears later needs a
    restart, which is the same contract as every other integration here."""
    if not _state["checked"]:
        _state["checked"] = True
        uri = _uri()
        if uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(EXPERIMENT)
                _state["enabled"] = True
                log.info("mlflow.enabled", uri=uri, experiment=EXPERIMENT)
            except Exception as exc:  # noqa: BLE001
                capture(exc, where="mlflow.init", uri=uri)
    return bool(_state["enabled"])


def _clean(d: dict | None) -> dict:
    """Only the scalars MLflow can store, and only the finite ones.

    A NaN reaches the server as a string and then sorts and plots as though it were a number, which is worse
    than the metric being absent.
    """
    import math

    out: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, int | float) and math.isfinite(v):
            out[k] = float(v)
    return out


def log_evaluation(*, model_version: str, gold_id: str, metrics: dict,
                   run_name: str | None = None, tags: dict | None = None) -> str | None:
    """Publish one evaluation. Returns the MLflow run id, or None when publishing is off or failed.

    Never raises. An evaluation that ran and a tracking server that did not is a reporting failure, and
    turning it into a promotion failure would make the gate depend on a system that has no business being in
    that decision.
    """
    if not enabled():
        return None
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name or f"{model_version}@{gold_id}") as run:
            mlflow.set_tags({
                "model_version": model_version,
                "gold_id": gold_id,
                # Where the weights came from. An external model scored on our gold is a different claim from
                # one this engine trained, and a history that conflates them is misleading.
                "source": (tags or {}).get("source", "labeloxav"),
                **{k: str(v) for k, v in (tags or {}).items()},
            })
            mlflow.log_params({
                "model_version": model_version,
                "gold_id": gold_id,
                # The population, beside the number, always.
                "gold_declared": metrics.get("gold_declared"),
                "gold_resolvable": metrics.get("gold_resolvable"),
                "gold_scored": metrics.get("prediction_plane_scored"),
            })
            scalars = _clean({
                k: metrics.get(k) for k in (
                    "map50", "map50_95", "precision", "recall", "safe_miou",
                    "prediction_plane_ap50", "harness_delta",
                    # A boolean, logged as a metric on purpose: it is the thing to filter and alert a history
                    # on, and a tag cannot be plotted over time.
                    "harness_divergent",
                )
            })
            if scalars:
                mlflow.log_metrics(scalars)
            for name, value in (metrics.get("per_class_recall") or {}).items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"recall/{name}", float(value))
            for name, value in (metrics.get("per_class_ap50") or {}).items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"ap50/{name}", float(value))
            log.info("mlflow.evaluation_logged", model_version=model_version, gold_id=gold_id,
                     run_id=run.info.run_id)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        capture(exc, where="mlflow.log_evaluation", model_version=model_version, gold_id=gold_id)
        return None


def log_promotion(*, model_version: str, promoted: bool, gate: dict,
                  gold_id: str | None = None) -> str | None:
    """Publish a promotion decision, including a refusal.

    The refusals are the more useful half. Every model in this corpus was blocked for months, first by a
    measurement fault and then on safety recall floors, and a history that recorded only successful
    promotions would show nothing at all for that whole period.
    """
    if not enabled():
        return None
    try:
        import mlflow

        with mlflow.start_run(run_name=f"gate:{model_version}") as run:
            mlflow.set_tags({
                "model_version": model_version, "kind": "promotion",
                "promoted": str(bool(promoted)),
                "gold_id": gold_id or gate.get("comparison_basis") or "",
                # Why it was refused, in the gate's own words, so a reader does not have to reconstruct it
                # from the metrics that happen to be present.
                "reasons": "; ".join(gate.get("reasons") or [])[:480],
            })
            mlflow.log_metrics(_clean({
                "promoted": promoted,
                "map_delta": gate.get("map_delta"),
                "beats_map": gate.get("beats_map"),
                "safety_ok": gate.get("safety_ok"),
                "recall_ok": gate.get("recall_ok"),
            }))
            log.info("mlflow.promotion_logged", model_version=model_version, promoted=promoted,
                     run_id=run.info.run_id)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        capture(exc, where="mlflow.log_promotion", model_version=model_version)
        return None


def status() -> dict:
    return {"enabled": enabled(), "tracking_uri": _uri(), "experiment": EXPERIMENT}
