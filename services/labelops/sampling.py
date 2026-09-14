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


def rogan_gladen_interval(observed_p: float, *, sens_ci: dict, spec_ci: dict) -> dict:
    """Correct a rate for an imperfect judge, carrying the judge's own uncertainty through.

    The point version collapses three uncertain quantities into one confident-looking number, and on real
    data that is worse than useless. Measured on this corpus the judge came out at sensitivity 0.76
    (0.65 to 0.84) and specificity 0.80 (0.65 to 0.90), which puts the estimator's denominator anywhere
    between 0.30 and 0.74. A denominator uncertain by a factor of two makes the corrected rate uncertain by
    a factor of two, and quoting its midpoint would hide exactly that.

    So the correction is evaluated at both ends of the judge's intervals. Note this is the range implied by
    the judge's uncertainty alone; the sampling error in `observed_p` is reported separately by the caller
    and the two are not combined, because combining them would imply a joint interval nobody computed.

    `clamped` is the important flag. Rogan-Gladen is unbounded, so a judge whose measured error cannot
    explain the observed rate produces an estimate above 1.0 or below 0.0, which then gets clipped into
    range and reads as a confident 1.0. That is a signal that the model does not fit, not an answer, and it
    has to be visible.
    """
    lo_est = rogan_gladen(observed_p, sensitivity=sens_ci["lo"], specificity=spec_ci["lo"])
    hi_est = rogan_gladen(observed_p, sensitivity=sens_ci["hi"], specificity=spec_ci["hi"])
    mid = rogan_gladen(observed_p, sensitivity=sens_ci["p"], specificity=spec_ci["p"])

    ends = [v for v in (lo_est, hi_est, mid) if v is not None]
    if not ends:
        return {"p": None, "lo": None, "hi": None, "clamped": False,
                "note": "the judge carries no information (sensitivity + specificity <= 1), so no "
                        "correction is possible at any point in its interval"}

    def _raw(sens: float, spec: float) -> float | None:
        d = sens + spec - 1.0
        return None if d <= 1e-9 else (observed_p + spec - 1.0) / d

    raws = [r for r in (_raw(sens_ci["lo"], spec_ci["lo"]), _raw(sens_ci["hi"], spec_ci["hi"]),
                        _raw(sens_ci["p"], spec_ci["p"])) if r is not None]
    clamped = any(r > 1.0 or r < 0.0 for r in raws)

    note = None
    if clamped:
        note = ("the corrected estimate falls outside [0, 1] before clamping, which means the observed rate "
                "is more extreme than this judge's measured error can explain. Treat it as a bound, not a "
                "point estimate: either the judge is better than its calibration suggests, or the "
                "calibration set is not representative of the batch being corrected")
    return {"p": round(mid, 4) if mid is not None else None,
            "lo": round(min(ends), 4), "hi": round(max(ends), 4),
            "clamped": clamped, "note": note}


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


# ---------------------------------------------------------------------------------------------------
# Sequential acceptance: Wald's SPRT over a stream of verdicts.
#
# A fixed draw asks the same number of questions of a clean class and a hopeless one. The SPRT asks
# until the evidence is decisive, which for a clean class is sooner than the fixed n and for a bad one
# is a handful of defects. It tests H0: p = p1 (good, the rate we are happy with) against H1: p = p0
# (bad, the far bound); every verdict moves a log-likelihood ratio by a fixed step and the decision is
# the first crossing of a bound. Nothing in here consults the database.
# ---------------------------------------------------------------------------------------------------


def sprt_bounds(*, alpha: float = 0.05, beta: float = 0.10) -> dict:
    """The two Wald bounds on the log-likelihood ratio of bad over good.

    Crossing `bound_reject` (upper) means the sample looks like the bad rate; crossing `bound_accept`
    (lower) means it looks like the good rate. Wald's approximations, which are conservative in the
    direction that matters here (true error rates are at or below alpha and beta)."""
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1)")
    return {"bound_accept": math.log(beta / (1 - alpha)),
            "bound_reject": math.log((1 - beta) / alpha)}


def sprt_steps(*, p0: float, p1: float) -> dict:
    """Per-verdict increments of the log-likelihood ratio: one for a defect, one for a clean object."""
    if not (0 < p1 < p0 < 1):
        raise ValueError("need 0 < p1 (good) < p0 (bad) < 1")
    return {"defect": math.log(p0 / p1), "clean": math.log((1 - p0) / (1 - p1))}


def sprt_llr(defects: int, n: int, *, p0: float, p1: float) -> float:
    steps = sprt_steps(p0=p0, p1=p1)
    return defects * steps["defect"] + (n - defects) * steps["clean"]


def sprt_oc(p: float, *, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.10) -> float:
    """Wald's operating characteristic: the probability the test ends in accept when the true rate is p.

    L(p) = (A^h - 1) / (A^h - B^h) where h solves p*(p0/p1)^h + (1-p)*((1-p0)/(1-p1))^h = 1; h is
    found by bisection, and the h -> 0 limit at the indifference point is taken analytically. Checks:
    L(p1) = 1 - alpha, L(p0) = beta."""
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    b = sprt_bounds(alpha=alpha, beta=beta)
    ln_a, ln_b = b["bound_reject"], b["bound_accept"]
    steps = sprt_steps(p0=p0, p1=p1)

    def g(h: float) -> float:
        # Exponents clamped: past 700 the term overflows a float, and by then the sign is settled.
        return (p * math.exp(min(700.0, h * steps["defect"]))
                + (1 - p) * math.exp(min(700.0, h * steps["clean"])) - 1)

    # g(0) = 0 always; the other root is the h we want. Its sign is that of -g'(0), which is the sign
    # of the expected step: positive expected drift (bad-looking p) gives h < 0. Past |h| = 500 the
    # root has effectively escaped (p at or beyond an extreme) and L is at its limit.
    drift = p * steps["defect"] + (1 - p) * steps["clean"]
    if abs(drift) < 1e-12:
        # Limit h -> 0: L = ln A / (ln A - ln B).
        return ln_a / (ln_a - ln_b)
    if drift < 0:
        lo, hi = 1e-9, 1.0
        while g(hi) < 0:
            hi *= 2
            if hi > 500:
                return 1.0
    else:
        lo, hi = -1.0, -1e-9
        while g(lo) < 0:
            lo *= 2
            if lo < -500:
                return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            if drift < 0:
                lo = mid
            else:
                hi = mid
        else:
            if drift < 0:
                hi = mid
            else:
                lo = mid
    h = (lo + hi) / 2
    a_h, b_h = math.exp(min(700.0, ln_a * h)), math.exp(min(700.0, ln_b * h))
    denom = a_h - b_h
    if abs(denom) < 1e-300:
        return ln_a / (ln_a - ln_b)
    return min(1.0, max(0.0, (a_h - 1) / denom))


def sprt_p_hat(defects: int, n: int, *, p0: float, p1: float) -> float:
    """Observed defect rate smoothed by one pseudo-verdict at the middle of the indifference zone."""
    return (defects + (p0 + p1) / 2) / (n + 1)


def sprt_expected_remaining(defects: int, n: int, *, p0: float, p1: float,
                            alpha: float = 0.05, beta: float = 0.10,
                            p_hat: float | None = None) -> float:
    """Wald's ASN from the current position: verdicts still expected before a bound is crossed.

    The true rate is unknown; the plug-in is the observed rate smoothed by one pseudo-verdict at the
    midpoint of the indifference zone, `(defects + (p0+p1)/2) / (n+1)`, unless a `p_hat` is given.
    A Laplace 0.5 pseudo-count would read an unjudged lot as half defective and rank it last on the
    worklist; the indifference prior reads it as undecided. Zero once a bound is already crossed."""
    b = sprt_bounds(alpha=alpha, beta=beta)
    z = sprt_llr(defects, n, p0=p0, p1=p1)
    if z <= b["bound_accept"] or z >= b["bound_reject"]:
        return 0.0
    if p_hat is None:
        p_hat = sprt_p_hat(defects, n, p0=p0, p1=p1)
    p_hat = min(1.0, max(0.0, p_hat))
    steps = sprt_steps(p0=p0, p1=p1)
    drift = p_hat * steps["defect"] + (1 - p_hat) * steps["clean"]
    oc = sprt_oc(p_hat, p0=p0, p1=p1, alpha=alpha, beta=beta)
    if abs(drift) < 1e-12:
        # Zero-drift ASN: E[Z^2 at stop] / E[step^2].
        second = p_hat * steps["defect"] ** 2 + (1 - p_hat) * steps["clean"] ** 2
        end = oc * (b["bound_accept"] - z) ** 2 + (1 - oc) * (b["bound_reject"] - z) ** 2
        return max(0.0, end / second)
    expected_end = oc * (b["bound_accept"] - z) + (1 - oc) * (b["bound_reject"] - z)
    return max(0.0, expected_end / drift)


def sprt_decision(defects: int, n: int, *, p0: float, p1: float,
                  alpha: float = 0.05, beta: float = 0.10) -> dict:
    """The sequential verdict after `defects` in `n` verdicts.

    `verdict` is `accept` (llr at or below the lower bound), `reject` (at or above the upper bound) or
    `continue`. `expected_remaining` is the ASN from here at the smoothed observed rate, and `oc` is
    the acceptance probability at that rate. The caller decides whether an accept is sufficient on its
    own; settlement requires the fixed-sample Wilson rule to agree."""
    if n < 0 or defects < 0 or defects > n:
        raise ValueError("defects must be in [0, n]")
    b = sprt_bounds(alpha=alpha, beta=beta)
    z = sprt_llr(defects, n, p0=p0, p1=p1)
    if z <= b["bound_accept"]:
        verdict = "accept"
    elif z >= b["bound_reject"]:
        verdict = "reject"
    else:
        verdict = "continue"
    p_hat = sprt_p_hat(defects, n, p0=p0, p1=p1)
    return {"llr": round(z, 4), "bound_accept": round(b["bound_accept"], 4),
            "bound_reject": round(b["bound_reject"], 4), "verdict": verdict,
            "p0": p0, "p1": p1, "alpha": alpha, "beta": beta, "n": n, "defects": defects,
            "p_hat": round(p_hat, 4),
            "oc": round(sprt_oc(p_hat, p0=p0, p1=p1, alpha=alpha, beta=beta), 4),
            "expected_remaining": round(sprt_expected_remaining(
                defects, n, p0=p0, p1=p1, alpha=alpha, beta=beta), 1),
            "reason": {"accept": f"llr {z:.2f} at or below {b['bound_accept']:.2f} after {n} verdicts",
                       "reject": f"llr {z:.2f} at or above {b['bound_reject']:.2f} after {n} verdicts",
                       "continue": f"llr {z:.2f} between the bounds after {n} verdicts"}[verdict]}
