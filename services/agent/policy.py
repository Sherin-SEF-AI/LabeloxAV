"""The accept/route policy: given an object's calibrated confidence, cross-model agreement, single-frame
quality verdict, and the self-consistency critic's verdict, decide the one autonomous action the agent is
allowed to take -- auto_accept -- or defer to a human (review / annotate).

Auto-accept is the only state the agent writes on its own; everything else routes work to a person. The
critic can only VETO an auto-accept (demote to review), never create one. This keeps the failure mode
one-directional: the worst the agent does autonomously is accept a wrong label, which the control-sample
review and the reversible AgentRun both catch. It never auto-rejects or auto-deletes.

The thresholds come from the gate's own configuration, resolved at construction, because a second copy
of them was this module's one serious defect. The docstring here used to say the defaults "mirror the
gate (0.95 ... 0.60)". They did not: configs/default.yaml runs the gate at 0.45 / 0.08 on a calibrated
confidence scale whose corpus p50 is 0.411 and p90 is 0.479, so this module's hardcoded 0.95 matched
26 objects out of 505,288 (0.005%) and its 0.60 floor routed 91.4% of everything to full re-annotation.
The agent-side flywheel was, structurally, a no-op that nobody had measured.

A dataclass rather than a bare config read, still, so a run can tighten a threshold on purpose (a fresh
ontology wants 0.98) and record exactly what it used - but the DEFAULT is now the gate's running value,
and there is no literal here to drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


def _gate_auto_accept() -> float:
    from core.config import get_settings

    return float(get_settings().gate.auto_accept)


def _gate_review_low() -> float:
    from core.config import get_settings

    return float(get_settings().gate.review_low)


@dataclass(frozen=True)
class PolicyThresholds:
    # Defaults resolve from the gate's running configuration at construction time, so the agent and the
    # batch pipeline cannot disagree about what "confident" means. Pass explicit values to override on
    # purpose; never to restate the config.
    auto_accept_conf: float = field(default_factory=_gate_auto_accept)
    review_low: float = field(default_factory=_gate_review_low)
    require_agreement: bool = True   # cross-path (model) agreement required to auto-accept

    def to_dict(self) -> dict:
        return asdict(self)

    def for_class(self, class_id: int, onto, fitted: dict[int, float] | None = None) -> PolicyThresholds:
        """These thresholds with the per-class auto-accept boundary the gate itself would use.

        Delegates to the gate's `class_auto_accept`, which prefers a measured ThresholdFit for the class
        and otherwise falls back to the configured constant (logging that nobody measured it). Safety
        classes get the stricter configured bound the same way the gate gives it to them. An explicitly
        overridden auto_accept_conf is respected: a caller who asked for 0.98 asked for 0.98 everywhere.
        """
        from core.config import get_settings
        from services.autolabel.gate import class_auto_accept

        if self.auto_accept_conf != _gate_auto_accept():
            return self   # explicit override wins over per-class resolution
        t = class_auto_accept(class_id, onto, get_settings().gate, fitted)
        return replace(self, auto_accept_conf=float(t))


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
