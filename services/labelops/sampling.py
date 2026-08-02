"""How many to check, and what the answer is worth once you have.

Every quality number this system reports is a bare point estimate. `measured_precision` returns a fraction,
`honeypot_accuracy` returns a ratio, the overnight auditor samples a hardcoded 200. None of them says how
sure it is, so "precision is 0.87" reads identically whether it came from 12 objects or 12,000, and a
customer buying a quality claim is buying the interval as much as the number.

Two things live here.

A Wilson interval, not the textbook normal approximation. At the rates that matter here, a defect rate near
0 or near 1 on a few hundred samples, the normal interval is wrong in the direction that flatters: it
produces bounds below zero for a clean batch and is far too narrow when p is extreme. Wilson stays inside
[0, 1] and holds its coverage at small n, which is the whole regime this corpus is in.

And a sample size, so "check some" becomes a number somebody can plan around.

**Sampling for precision is not sampling for improvement, and the difference matters.** The active-learning
queue deliberately surfaces the hardest, most uncertain objects, which is right for teaching the model and
ruinous for measuring it: judging that batch tells you the accuracy of the worst objects in the corpus, not
of the corpus. A precision estimate needs a sample that is random with respect to correctness.
"""

from __future__ import annotations

import math

# The z for a two-sided interval at a given confidence. Tabulated rather than pulling in scipy for three
# constants, and the ones nobody uses are omitted rather than approximated.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> dict:
    """A proportion with its uncertainty. Returns {p, lo, hi, n, half_width}.

    n = 0 gives the whole interval rather than an error, because "we have not checked any" is a real state
    and reporting it as 0.0 precision would be a lie in the confident direction.
    """
    if n <= 0:
        return {"p": None, "lo": 0.0, "hi": 1.0, "n": 0, "half_width": 1.0,
                "note": "nothing sampled yet, so the rate is unknown rather than zero"}
    z = _Z.get(round(confidence, 2), 1.96)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - margin), min(1.0, centre + margin)
    return {"p": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n,
            "half_width": round((hi - lo) / 2, 4)}


def sample_size_for(half_width: float, *, expected_p: float = 0.5, confidence: float = 0.95) -> int:
    """How many to check to pin a rate to within +/- half_width.

    `expected_p` defaults to 0.5 because that is the worst case and therefore the safe planning assumption.
    Supplying a better guess from a pilot shrinks the requirement sharply: a rate near 0.9 needs about a
    third of what 0.5 does.
    """
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    z = _Z.get(round(confidence, 2), 1.96)
    n = (z * z * expected_p * (1 - expected_p)) / (half_width * half_width)
    return int(math.ceil(n))


def rogan_gladen(observed_p: float, *, sensitivity: float, specificity: float) -> float | None:
    """Correct a rate measured by an imperfect judge for that judge's own error.

    The reason this is needed rather than optional. A VLM can judge 570,379 labels; a person cannot. But the
    judge is wrong sometimes, and quoting its raw agreement rate as precision embeds its error in every
    number downstream, in an unknown direction. If the judge is 90% sensitive and calls 85% of labels
    correct, the true rate is not 85%.

    Measure the judge against a human-adjudicated subsample to get its sensitivity (it says correct when the
    label is correct) and specificity (it says incorrect when the label is wrong), then invert:

        p_true = (p_observed + specificity - 1) / (sensitivity + specificity - 1)

    the standard Rogan-Gladen prevalence estimator. Returns None when sensitivity + specificity <= 1, which
    means the judge carries no information (at exactly 1 it is a coin, below it is anti-correlated) and no
    correction can recover a rate from it. That is a real state and worth refusing to answer for, since the
    formula happily returns a confident-looking number either side of the singularity.

    Clamped to [0, 1]: sampling noise in a small subsample can push the estimate outside the range it is
    estimating, and a precision of 1.04 is less useful than a precision of 1.0 with a wide interval.
    """
    denom = sensitivity + specificity - 1.0
    if denom <= 1e-9:
        return None
    return max(0.0, min(1.0, (observed_p + specificity - 1.0) / denom))


def acceptance_decision(defects: int, n: int, *, max_defect_rate: float,
                        confidence: float = 0.95) -> dict:
    """Accept a batch, reject it, or say the sample is too small to tell.

    The third answer is the one that matters and the one a bare threshold cannot give. A batch of 20 with one
    defect has an observed rate of 0.05 and an upper bound near 0.25, so calling it acceptable against a 10%
    limit is a statement the evidence does not support. Judged on the interval rather than the point
    estimate: accept only when the upper bound clears the limit, reject only when the lower bound exceeds it,
    and otherwise say so and ask for more.
    """
    ci = wilson_interval(defects, n, confidence)
    if n <= 0:
        return {**ci, "verdict": "unknown", "reason": "nothing sampled"}
    if ci["hi"] <= max_defect_rate:
        verdict, reason = "accept", (f"defect rate is at most {ci['hi']:.1%} with {confidence:.0%} "
                                     f"confidence, within the {max_defect_rate:.1%} limit")
    elif ci["lo"] > max_defect_rate:
        verdict, reason = "reject", (f"defect rate is at least {ci['lo']:.1%}, above the "
                                     f"{max_defect_rate:.1%} limit")
    else:
        need = sample_size_for(max(0.01, max_defect_rate / 2), expected_p=ci["p"] or 0.5,
                               confidence=confidence)
        verdict, reason = "inconclusive", (
            f"observed {ci['p']:.1%} but the interval spans the {max_defect_rate:.1%} limit "
            f"({ci['lo']:.1%} to {ci['hi']:.1%}); about {need} samples would settle it")
    return {**ci, "verdict": verdict, "reason": reason, "max_defect_rate": max_defect_rate}
