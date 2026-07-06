"""Per-plane SLOs + observability (M19). Each data-engine plane declares the objectives it must meet: a latency
budget, an error-rate ceiling, a coverage floor. This evaluates a plane's measured metrics against its declared
SLO and reports which objectives breached, so a fleet-scale operator sees the bottleneck plane directly rather
than reading scattered logs. Pure over the measurements, so the objective evaluation is testable; the ledger
write lives in run.py."""

from __future__ import annotations

# per-plane service-level objectives. op ">=" means the metric must be at least the threshold (a floor);
# op "<=" means it must be at most the threshold (a ceiling).
PLANE_SLOS = {
    "sanyx":   [{"metric": "ingest_p95_s", "op": "<=", "threshold": 300.0},
                {"metric": "gate_error_rate", "op": "<=", "threshold": 0.01}],
    "calyx":   [{"metric": "calibration_p95_s", "op": "<=", "threshold": 120.0},
                {"metric": "recover_success_rate", "op": ">=", "threshold": 0.80}],
    "sievyx":  [{"metric": "cluster_p95_s", "op": "<=", "threshold": 60.0},
                {"metric": "odd_coverage", "op": ">=", "threshold": 0.90}],
    "labelox": [{"metric": "queue_p95_s", "op": "<=", "threshold": 2.0},
                {"metric": "annotation_quality_mean", "op": ">=", "threshold": 0.85}],
    "oraclyx": [{"metric": "fusion_p95_s", "op": "<=", "threshold": 30.0},
                {"metric": "consensus_rate", "op": ">=", "threshold": 0.70}],
    "verdyx":  [{"metric": "eval_p95_s", "op": "<=", "threshold": 600.0},
                {"metric": "safety_recall", "op": ">=", "threshold": 0.50}],
    "forgyx":  [{"metric": "compile_p95_s", "op": "<=", "threshold": 900.0},
                {"metric": "deploy_success_rate", "op": ">=", "threshold": 0.95}],
}


def _satisfies(value: float, op: str, threshold: float) -> bool:
    return value <= threshold if op == "<=" else value >= threshold


def evaluate_slos(plane: str, measurements: dict) -> dict:
    """Evaluate a plane's measured metrics against its declared SLO. A metric that is missing from the
    measurements is itself a breach (unobservable is not assumed healthy). Returns whether every objective was
    met and the list of breaches with the measured value and the threshold it violated."""
    objectives = PLANE_SLOS.get(plane)
    if objectives is None:
        raise ValueError(f"unknown plane {plane}")
    breaches = []
    for obj in objectives:
        m, op, thr = obj["metric"], obj["op"], obj["threshold"]
        if m not in measurements or measurements[m] is None:
            breaches.append({"metric": m, "value": None, "threshold": thr, "op": op, "reason": "unobserved"})
            continue
        val = float(measurements[m])
        if not _satisfies(val, op, thr):
            breaches.append({"metric": m, "value": val, "threshold": thr, "op": op, "reason": "breached"})
    return {"plane": plane, "met": len(breaches) == 0, "breaches": breaches,
            "n_objectives": len(objectives)}
