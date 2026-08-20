"""Which detections are worth a person's time, measured by what the model is torn between.

Confidence alone cannot answer this. Two detections at 0.55 are not equally informative: one is a scooter
the model is simply unsure about, and more scooters in bad light will fix it; the other is a model split
evenly between scooter and motorcycle, and no amount of extra scooters will fix it because the problem is
where the boundary sits. Only the second is worth a label, and `class_id` plus `conf` cannot tell them
apart, which is why db/models.py::Prediction gained `class_probs`.

The margin is the gap between the top two probabilities. A small margin means the model is deciding
between two classes rather than doubting one, and the pair it is deciding between says which decision
boundary a label would buy.

WEIGHTED BY WHAT THE CONFUSION COSTS. A scooter called a motorcycle and a pedestrian called a bollard are
both confusions and are not both worth the same. The score multiplies the ambiguity by the pack's cost for
that ordered pair, so a tight margin inside a cheap clique ranks below a looser one that crosses a safety
boundary. Without that weighting active learning spends its whole budget on two-wheelers, which is where
the ambiguity is densest and the cost is lowest.

REFUSING IS PART OF THE JOB. A prediction with no distribution (every one written before `class_probs`
existed, and any model whose runtime does not expose per-class scores) scores None rather than zero. Zero
would sort it to the bottom alongside the confident ones, which is a claim about it that nothing supports.

NumPy is the oracle; the torch path mirrors the scoring and must agree to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass(frozen=True)
class MarginScore:
    """One detection's claim on a labelling budget.

    `margin` is p1 - p2 over the class distribution: small means torn. `score` is the weighted ambiguity,
    (1 - margin) * pair_cost, so a confident detection scores near zero however expensive its class pair
    would have been. `top_pair` names what it is torn between, which is what makes a batch reviewable:
    "these 40 frames are the scooter/motorcycle boundary" is actionable and "these 40 frames are uncertain"
    is not.

    `measured` is False, with a reason, when the prediction carries no distribution at all.
    """

    measured: bool
    margin: float | None
    score: float | None
    top_pair: tuple[int, int] | None
    top_probs: tuple[float, float] | None
    clique: str | None
    pair_cost: float | None
    reason: str | None = None


def _unmeasured(reason: str) -> MarginScore:
    return MarginScore(measured=False, margin=None, score=None, top_pair=None, top_probs=None,
                       clique=None, pair_cost=None, reason=reason)


def margin_score(class_probs: dict[Any, Any] | None, *, pair_cost: Any = None,
                 clique_of: Any = None) -> MarginScore:
    """Score one prediction's distribution. `class_probs` maps class id (as str or int) to probability."""
    if not class_probs:
        return _unmeasured("no class distribution was stored for this prediction")
    try:
        items = sorted(((int(k), float(v)) for k, v in class_probs.items()), key=lambda kv: -kv[1])
    except (TypeError, ValueError):
        return _unmeasured("the stored class distribution is not a class-to-probability map")
    if len(items) < 2:
        # One class in the distribution means nothing was competing with it. That is a real answer, not a
        # missing one: the margin is the full width and the detection is not a boundary case.
        return MarginScore(measured=True, margin=1.0, score=0.0, top_pair=None,
                           top_probs=(items[0][1], 0.0), clique=None, pair_cost=None,
                           reason="only one class had any probability mass")

    (a, pa), (b, pb) = items[0], items[1]
    margin = pa - pb
    cost = float(pair_cost(a, b)) if pair_cost is not None else 1.0
    clique = clique_of(a, b) if clique_of is not None else None
    return MarginScore(measured=True, margin=round(margin, 6),
                       score=round((1.0 - margin) * cost, 6), top_pair=(a, b),
                       top_probs=(round(pa, 6), round(pb, 6)), clique=clique,
                       pair_cost=round(cost, 6))


def _scores_np(p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64],
               costs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (1.0 - (p1 - p2)) * costs


def _scores_torch(p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64],
                  costs: npt.NDArray[np.float64],
                  device: str) -> npt.NDArray[np.float64]:  # pragma: no cover - GPU path
    t1 = torch.as_tensor(p1, device=device, dtype=torch.float64)
    t2 = torch.as_tensor(p2, device=device, dtype=torch.float64)
    tc = torch.as_tensor(costs, device=device, dtype=torch.float64)
    out: npt.NDArray[np.float64] = ((1.0 - (t1 - t2)) * tc).cpu().numpy()
    return out


def score_batch(distributions: list[dict[Any, Any] | None], *, pair_cost: Any = None, clique_of: Any = None,
                device: str | None = None) -> dict[str, Any]:
    """Score many predictions at once. Returns {"scores", "measured", "detail", "n_unmeasured"}.

    `scores` is a float array with NaN where the prediction carried no distribution, so the caller sorts
    with `np.argsort` on a masked view rather than being handed a zero it cannot distinguish from a
    genuinely confident detection.
    """
    detail = [margin_score(d, pair_cost=pair_cost, clique_of=clique_of) for d in distributions]
    n = len(detail)
    p1 = np.zeros(n, dtype=np.float64)
    p2 = np.zeros(n, dtype=np.float64)
    costs = np.ones(n, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for i, d in enumerate(detail):
        if not d.measured or d.top_probs is None:
            continue
        ok[i] = True
        p1[i], p2[i] = d.top_probs
        costs[i] = d.pair_cost if d.pair_cost is not None else 1.0

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        raw = _scores_torch(p1, p2, costs, str(device))  # pragma: no cover - GPU path
    else:
        raw = _scores_np(p1, p2, costs)
    scores = np.where(ok, raw, np.nan)
    return {"scores": scores, "measured": ok, "detail": tuple(detail),
            "n_unmeasured": int((~ok).sum())}


__all__ = ["MarginScore", "margin_score", "score_batch"]
