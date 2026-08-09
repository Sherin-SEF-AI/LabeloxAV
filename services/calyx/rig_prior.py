"""A rig prior built from every session a vehicle has driven, and an honest account of what it rests on.

CALYX could already fuse per-session calibrations into a prior and score a drift, but nothing had ever run
either one over the corpus. Doing that first changed what was worth building, because the corpus answers a
question the fusion code does not ask.

All 101 calibrations carry source='estimated' and quality=0.6. Ninety-seven of them belong to one vehicle.
Every one of them has fx=fy=2870, cx=960, cy=540, xyz=[0, 0, 1.5], roll=0 and yaw=0. Only pitch differs
between sessions. So four of the six extrinsic degrees of freedom are not estimates at all, they are a
constant written once and copied.

That matters because the existing fusion reports agreement as a spread, and a spread of exactly zero reads as
perfect agreement across ninety-seven independent sessions. It is the strongest possible claim, produced by
the weakest possible evidence, and any consumer weighting geometry by it would be weighting a default. This
is the same failure the gold set and the two eval harnesses each turned out to have: a number computed over
nothing, presented like a number computed over everything.

So a prior here reports per axis whether the axis was measured at all, and refuses to express a deviation in
sigmas on an axis whose sigma came from constants. The scale is a median absolute deviation rather than a
standard deviation, because with a handful of badly-calibrated sessions in the pool a non-robust scale is set
by the outliers it is supposed to find.
"""

from __future__ import annotations

import numpy as np

# Below this, a per-axis spread is not small, it is absent: the values are byte-identical because they were
# filled in rather than measured.
DEGENERATE_SPREAD = 1e-9

# MAD to sigma for a normal distribution. Stated rather than inlined because the 1.4826 shows up in reviews
# as a magic number every time.
MAD_TO_SIGMA = 1.4826

# How many robust sigmas from the prior before a session's axis is called an outlier. Chosen to match the
# gate's habit elsewhere of flagging rather than blocking: this marks a session for a look, it does not
# discard its labels.
OUTLIER_SIGMA = 3.5

AXES_RPY = ("roll", "pitch", "yaw")
AXES_XYZ = ("x", "y", "z")


def _axis_stat(values: np.ndarray) -> dict:
    """Robust centre and scale for one axis, plus whether the axis carries any information at all."""
    v = np.asarray(values, dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    spread = float(np.max(v) - np.min(v)) if v.size else 0.0
    measured = spread > DEGENERATE_SPREAD
    return {
        "median": round(med, 6),
        # A degenerate axis gets no sigma. Reporting 0.0 would invite a division that makes every session
        # infinitely deviant, and reporting a floor would invent a tolerance nobody measured.
        "sigma": round(mad * MAD_TO_SIGMA, 6) if measured else None,
        "range": round(spread, 6),
        "measured": measured,
        "n": int(v.size),
    }


def build_rig_prior(calibs: list[dict]) -> dict:
    """Fuse a vehicle's per-session calibrations into a prior, per axis.

    `calibs` are dicts with `rpy_deg` (3) and `xyz_m` (3). Unlike `fuse_calibrations`, which returns one
    scalar spread over all of rpy at once, this keeps the axes apart, because on this corpus they are not
    alike: pitch is measured and the other five are constants, and a single pooled number hides that
    completely.
    """
    rows = [c for c in calibs if c.get("rpy_deg") and c.get("xyz_m")]
    if not rows:
        return {"n": 0, "axes": {}, "measured_axes": [], "constant_axes": [],
                "detail": "no calibrations with extrinsics"}

    rpy = np.array([c["rpy_deg"] for c in rows], dtype=np.float64)
    xyz = np.array([c["xyz_m"] for c in rows], dtype=np.float64)

    axes: dict[str, dict] = {}
    for i, name in enumerate(AXES_RPY):
        axes[name] = {**_axis_stat(rpy[:, i]), "unit": "deg"}
    for i, name in enumerate(AXES_XYZ):
        axes[name] = {**_axis_stat(xyz[:, i]), "unit": "m"}

    measured = [k for k, a in axes.items() if a["measured"]]
    constant = [k for k, a in axes.items() if not a["measured"]]
    return {
        "n": len(rows),
        "axes": axes,
        "measured_axes": measured,
        "constant_axes": constant,
        # Said in words because the shape of this corpus is the finding, and a caller that only reads the
        # numbers will otherwise treat five defaults as five agreements.
        "detail": (f"{len(measured)} of 6 extrinsic axes vary across {len(rows)} sessions "
                   f"({', '.join(measured) or 'none'}); "
                   f"{', '.join(constant) or 'none'} are identical in every session and carry no evidence"),
    }


def deviations(calib: dict, prior: dict) -> dict:
    """How far one session sits from the prior, per axis.

    A measured axis gets a deviation in robust sigmas and can be an outlier. A constant axis gets its
    absolute difference and nothing more: with no observed variation there is no scale to divide by, so any
    non-zero difference is merely different, not anomalous, and calling it an outlier would be arithmetic
    dressed as evidence.
    """
    if not prior.get("axes"):
        return {"outlier": False, "axes": {}, "detail": "no prior to compare against"}

    values = dict(zip(AXES_RPY, calib.get("rpy_deg") or [], strict=False))
    values.update(dict(zip(AXES_XYZ, calib.get("xyz_m") or [], strict=False)))

    out: dict[str, dict] = {}
    worst = 0.0
    for name, stat in prior["axes"].items():
        if name not in values:
            continue
        diff = float(values[name]) - float(stat["median"])
        entry: dict = {"delta": round(diff, 6), "unit": stat["unit"], "measured": stat["measured"]}
        sigma = stat["sigma"]
        if stat["measured"] and sigma and sigma > 0:
            z = abs(diff) / sigma
            entry["sigmas"] = round(z, 3)
            entry["outlier"] = bool(z >= OUTLIER_SIGMA)
            worst = max(worst, z)
        else:
            # No scale, so no verdict. Saying so beats a null that a caller will read as "fine".
            entry["sigmas"] = None
            entry["outlier"] = False
            entry["note"] = "axis is constant across the fleet, so a deviation cannot be scored"
        out[name] = entry

    flagged = [k for k, v in out.items() if v.get("outlier")]
    return {
        "outlier": bool(flagged),
        "axes": out,
        "worst_sigmas": round(worst, 3),
        "flagged_axes": flagged,
        "detail": (f"{', '.join(flagged)} beyond {OUTLIER_SIGMA} sigma" if flagged
                   else f"within {OUTLIER_SIGMA} sigma of the fleet prior on every measured axis"),
    }


def prior_confidence(prior: dict) -> float:
    """A single confidence for the prior, discounted by how little of it was actually measured.

    Existing callers want one number. The honest version of that number cannot be built from agreement alone,
    because agreement across constants is free. So it is the sample-size term scaled by the fraction of axes
    that carry any evidence: a prior over ninety-seven sessions that only ever measured pitch tops out at
    one sixth of the confidence it would otherwise claim.
    """
    n = int(prior.get("n", 0))
    if n <= 0:
        return 0.0
    total_axes = len(prior.get("axes") or {}) or 6
    measured = len(prior.get("measured_axes") or [])
    size_term = n / (n + 3.0)
    return round(float(size_term * (measured / total_axes)), 4)
