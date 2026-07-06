"""ORACLYX uncertainty-calibrated pseudo-GT (M14). A pseudo label is not a hard truth; distillation should
weight it by how much ORACLYX actually trusts it. This turns the raw agreement signals (consensus strength,
how many views saw it, calibration confidence, geometric depth uncertainty) into one calibrated uncertainty in
[0, 1] and a soft-target weight the training set carries per label. It also ranks the labels ORACLYX is least
sure about by expected information gain, so the human review budget lands on the pseudo labels whose correction
would teach the model the most. Pure functions, so the calibration monotonicity and the ranking are testable."""

from __future__ import annotations


def pseudo_label_uncertainty(consensus_score: float, n_views: int = 1, calib_confidence: float = 1.0,
                             depth_uncertainty: float = 0.0) -> float:
    """Calibrated uncertainty in [0, 1] for one pseudo label. Rises as consensus weakens, as fewer views saw
    the object, as calibration confidence drops, and as the geometric depth grows uncertain. Monotonic in every
    argument, so a strictly weaker signal never reports lower uncertainty."""
    c = max(0.0, min(1.0, consensus_score))
    calib = max(0.0, min(1.0, calib_confidence))
    depth_u = max(0.0, min(1.0, depth_uncertainty))
    view_support = 1.0 - 1.0 / (1.0 + max(0, n_views))   # 0 views -> 0, grows and saturates toward 1
    # combine as the complement of the joint "trust" of the independent signals
    trust = c * calib * (0.4 + 0.6 * view_support) * (1.0 - 0.5 * depth_u)
    return round(max(0.0, min(1.0, 1.0 - trust)), 4)


def soft_target(uncertainty: float, floor: float = 0.05) -> float:
    """The distillation weight for a pseudo label: certain labels weigh ~1, uncertain labels are damped toward
    the floor rather than dropped, so the model still sees them softly."""
    u = max(0.0, min(1.0, uncertainty))
    return round(max(floor, 1.0 - u), 4)


def information_gain(uncertainty: float, rarity: float = 0.0, disagreement: float = 0.0,
                     safety_weight: float = 1.0) -> float:
    """Expected training value of a human reviewing this pseudo label. Highest when ORACLYX is uncertain, the
    class is rare, the paths disagree, and the object is safety-relevant. Labels ORACLYX is already sure about
    score near zero (reviewing them teaches little)."""
    u = max(0.0, min(1.0, uncertainty))
    r = max(0.0, min(1.0, rarity))
    d = max(0.0, min(1.0, disagreement))
    base = 0.5 * u + 0.3 * d + 0.2 * r
    return round(base * max(0.0, safety_weight), 4)


def rank_disagreements(items: list[dict]) -> list[dict]:
    """Rank pseudo labels for human review by descending expected information gain. Each item carries at least
    ``uncertainty``; optional ``rarity``, ``disagreement``, ``safety_weight`` refine the score. Returns the
    items with an added info_gain, most-valuable first (the M14 disagreement queue)."""
    scored = []
    for it in items:
        g = information_gain(it.get("uncertainty", 0.0), it.get("rarity", 0.0),
                             it.get("disagreement", 0.0), it.get("safety_weight", 1.0))
        scored.append({**it, "info_gain": g})
    scored.sort(key=lambda x: x["info_gain"], reverse=True)
    return scored
