"""Cloud GPU cost control (M18). The hybrid A100 burst is real money; a runaway or mis-estimated job must be
refused before it is dispatched, not discovered on the bill. This gates a cloud job against two ceilings: a
hard per-job cap (no single job may exceed it) and the remaining budget in the current window (the job must fit
under what is left of the daily/session cap). Pure over the estimate and the running spend, so the ceiling
logic is testable; the actual dispatch and spend accounting stay in the cloud runbook."""

from __future__ import annotations


def estimate_job_cost(gpu_hours: float, hourly_usd: float) -> float:
    """The estimated cost of a job from its GPU-hours and the target's hourly rate."""
    return round(max(0.0, gpu_hours) * max(0.0, hourly_usd), 4)


def cost_gate(est_cost_usd: float, per_job_cap_usd: float, spent_usd: float, window_cap_usd: float) -> dict:
    """Admit or refuse a cloud job on cost.

    Refuses when the estimate exceeds the hard per-job cap, or when it would push the window spend past the
    window cap. Returns the decision, the remaining budget after the job if admitted, and the binding reason so
    the refusal is explainable (not a silent drop)."""
    remaining = window_cap_usd - spent_usd
    if est_cost_usd > per_job_cap_usd:
        return {"allowed": False, "reason": f"estimate ${est_cost_usd:.2f} exceeds per-job cap "
                f"${per_job_cap_usd:.2f}", "est_cost_usd": est_cost_usd, "remaining_after": round(remaining, 4)}
    if est_cost_usd > remaining:
        return {"allowed": False, "reason": f"estimate ${est_cost_usd:.2f} exceeds ${remaining:.2f} remaining "
                f"of the ${window_cap_usd:.2f} window", "est_cost_usd": est_cost_usd,
                "remaining_after": round(remaining, 4)}
    return {"allowed": True, "reason": None, "est_cost_usd": est_cost_usd,
            "remaining_after": round(remaining - est_cost_usd, 4)}
