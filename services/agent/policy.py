"""The accept/route policy: given an object's calibrated confidence, cross-model agreement, single-frame
quality verdict, and the self-consistency critic's verdict, decide the one autonomous action the agent is
allowed to take -- auto_accept -- or defer to a human (review / annotate).

Auto-accept is the only state the agent writes on its own; everything else routes work to a person. The
critic can only VETO an auto-accept (demote to review), never create one. This keeps the failure mode
one-directional: the worst the agent does autonomously is accept a wrong label, which the control-sample
review and the reversible AgentRun both catch. It never auto-rejects or auto-deletes.

Thresholds mirror the gate (0.95 calibrated auto-accept boundary, 0.60 review floor) so the agent and the
batch autolabel pipeline speak the same language; they are surfaced here as a dataclass so a run can tune
them (e.g. a stricter 0.98 for a fresh ontology) and record exactly what it used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PolicyThresholds:
    auto_accept_conf: float = 0.95   # calibrated confidence required to auto-accept
    review_low: float = 0.60         # below this, a full human annotate regardless of anything else
    require_agreement: bool = True   # cross-path (model) agreement required to auto-accept

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    action: str      # auto_accept | review | annotate
    reason: str      # short human-readable justification
    tier: str        # sure | review | uncertain  (for display grouping)


def decide(
    conf: float,
    agreement: bool,
    quality_ok: bool,
    critic_ok: bool,
    th: PolicyThresholds,
) -> Decision:
    """The single decision rule. Order matters: a hard floor first, then the two vetoes (single-frame
    quality, then the cross-frame/cross-modal critic), then the positive auto-accept test, else review."""
    if conf < th.review_low:
        return Decision("annotate", f"confidence {conf:.2f} below review floor {th.review_low:.2f}", "uncertain")
    if not quality_ok:
        return Decision("review", "single-frame quality reviewer demoted it", "review")
    if not critic_ok:
        return Decision("review", "failed the self-consistency critic", "review")
    if conf >= th.auto_accept_conf and (agreement or not th.require_agreement):
        why = f"calibrated confidence {conf:.2f} >= {th.auto_accept_conf:.2f}"
        if th.require_agreement:
            why += " with cross-model agreement"
        return Decision("auto_accept", why, "sure")
    if conf >= th.auto_accept_conf and not agreement:
        return Decision("review", "high confidence but no cross-model agreement (single-path)", "review")
    return Decision("review", f"confidence {conf:.2f} in the review band", "review")
