"""The annotation agent: an orchestration + audit + reversibility layer over the existing perception,
gating, calibration, and active-learning primitives. It does not re-implement detection or the gate; it
sequences them, cross-checks their output with a self-consistency critic, applies a calibrated
accept/route policy, and records every autonomous change as a reversible AgentRun. The whole point is that
a human supervises exceptions instead of driving every click, and can revert any run exactly.
"""
