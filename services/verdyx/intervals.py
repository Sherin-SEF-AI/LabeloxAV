"""Every rate this system reports, with the uncertainty it actually has.

`evaluate_gold_patches` returns `precision: 0.334` and `recall: 0.556`. On the DashLab detector those were
computed over nine matched objects. As point estimates they are indistinguishable from the same numbers
measured over nine thousand, and the promotion gate compares them as though they were.

That is not hypothetical here. The gate refused a challenger for "does not beat champion mAP (0.142 vs
0.169)" on a gold set where a single object moves mAP by roughly ten points. The refusal may well be right;
what is wrong is that nothing in the decision could tell a real regression from noise.

**Wilson, not normal-approximation.** The textbook `p +- 1.96*sqrt(p(1-p)/n)` is wrong in exactly this
corpus's situation: at small n it produces bounds outside [0, 1], and at p = 0 or p = 1 it collapses to zero
width, so a class with 0 of 6 recalled would report "0.000, no uncertainty" when the truth is that almost
nothing is known. Wilson stays inside the interval and keeps a sensible width at the edges, which is where
this data lives.

**The sample size is the deliverable.** A buyer is not purchasing 0.87; they are purchasing 0.87 +- 0.04 at
n = 180, and the answer to "what would +- 0.02 cost" is a number this module can give.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 95% two-sided. Kept as a constant rather than a parameter threaded everywhere, because a report mixing
# confidence levels is worse than one at a level somebody disagrees with.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A rate and what is known about it."""

    point: float
    low: float
    high: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def margin(self) -> float:
        """Half-width: the "+- 0.04" a reader expects beside a rate."""
        return self.width / 2.0

    def as_dict(self) -> dict:
        return {"value": round(self.point, 4), "low": round(self.low, 4),
                "high": round(self.high, 4), "n": self.n,
                "margin": round(self.margin, 4)}

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] n={self.n}"


def wilson(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    n = 0 returns the whole range rather than an error or a zero. Nothing was measured, and [0, 1] is the
    honest statement of that; a 0.0 with no interval would read as a measured failure.
    """
    if n <= 0:
        return Interval(point=0.0, low=0.0, high=1.0, n=0)
    successes = max(0, min(int(successes), int(n)))
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return Interval(point=p, low=max(0.0, centre - spread), high=min(1.0, centre + spread), n=n)


def required_n(margin: float, p: float = 0.5, z: float = Z_95) -> int:
    """How many labelled instances a target margin needs.

    p defaults to 0.5 because that is the worst case: it maximises the variance, so the answer is the number
    that suffices whatever the true rate turns out to be. Passing a known approximate rate gives the smaller,
    honest number for that specific case.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    p = min(max(p, 0.0), 1.0)
    return int(math.ceil((z * z * p * (1 - p)) / (margin * margin)))


def separated(a: Interval, b: Interval) -> bool:
    """Whether two rates are distinguishable at all, rather than which is larger.

    Overlapping intervals do not prove equality, and this deliberately does not claim they do. What it says
    is weaker and more useful to a gate: on this much evidence, calling one better than the other is not
    supported.
    """
    return a.low > b.high or b.low > a.high


def compare(a: Interval, b: Interval) -> dict:
    """A difference between two rates, with whether the evidence supports calling it one.

    `decisive` false does not mean the models are equal. It means this sample cannot tell them apart, which
    is the thing a promotion decision needs to know and a bare delta hides.
    """
    return {
        "a": a.as_dict(), "b": b.as_dict(),
        "delta": round(a.point - b.point, 4),
        "decisive": separated(a, b),
        "detail": (f"{a.point:.3f} vs {b.point:.3f} at n={a.n} and n={b.n}: "
                   + ("the intervals do not overlap" if separated(a, b)
                      else "the intervals overlap, so this sample cannot separate them")),
    }


def from_counts(*, tp: int, fp: int, fn: int) -> dict:
    """Precision and recall as intervals, from the counts an evaluation already produces.

    Precision is over predictions (tp + fp) and recall over ground truth (tp + fn). They routinely have
    different denominators, which is exactly why a single "n" beside both is misleading and each interval
    carries its own.
    """
    return {
        "precision": wilson(tp, tp + fp).as_dict(),
        "recall": wilson(tp, tp + fn).as_dict(),
    }


def annotate_per_class(matched: dict[str, int], totals: dict[str, int]) -> dict:
    """Per-class recall as intervals.

    The per-class numbers are where small samples bite hardest: a class with six gold instances reports
    recall to three decimal places while its interval spans most of the range. The safety floors in the
    promotion gate are applied per class, so this is precisely where a point estimate can block or pass a
    model on almost no evidence.
    """
    out: dict[str, dict] = {}
    for name, n in totals.items():
        if not n:
            continue
        out[name] = wilson(int(matched.get(name, 0)), int(n)).as_dict()
    return out
