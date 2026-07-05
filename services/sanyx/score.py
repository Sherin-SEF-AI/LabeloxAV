"""SANYX health scoring: fold the six check results into one 0..100 score and a decision in
{pass, degraded, quarantine}. The decision is the gate the whole engine keys off: a quarantined session
never reaches SIEVYX or the label queue, so bad data never costs a label.

Two ways a session quarantines: the weighted score falls below the quarantine floor, OR any single check
hard-fails (returns status == "quarantine", e.g. a lost time-sync pulse). A safety-critical check can thus
veto an otherwise-fine score."""

from __future__ import annotations

from core.config import SanyxSettings

_RANK = {"pass": 0, "degraded": 1, "quarantine": 2}


def aggregate(checks: list[dict], cfg: SanyxSettings) -> dict:
    """Combine per-check results into an overall HealthReport payload: score (0..100), decision, and the
    checks with their sub-scores preserved."""
    if not checks:
        return {"score": 0.0, "decision": "quarantine", "checks": [], "verdict": "fail"}

    weights = cfg.weights
    total_w = 0.0
    acc = 0.0
    for c in checks:
        w = float(weights.get(c["name"], 0.0))
        total_w += w
        acc += w * float(c.get("score", 0.0))
    score = round(100.0 * acc / total_w, 1) if total_w > 0 else 0.0

    hard_fail = any(c.get("status") == "quarantine" for c in checks)
    if hard_fail or score < cfg.quarantine_below:
        decision = "quarantine"
    elif score >= cfg.pass_min:
        decision = "pass"
    else:
        decision = "degraded"

    # verdict mirrors the existing session_health tri-state (pass|warn|fail) so the older M-I.2 readers keep working
    verdict = {"pass": "pass", "degraded": "warn", "quarantine": "fail"}[decision]
    return {
        "score": score,
        "decision": decision,
        "verdict": verdict,
        "hard_failed": [c["name"] for c in checks if c.get("status") == "quarantine"],
        "checks": checks,
    }
