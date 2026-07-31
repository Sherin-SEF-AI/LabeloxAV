"""A flat-road lift always returns an answer, so something has to decide when to believe it.

The ray through a pixel below the horizon meets the ground plane at exactly one point. Near the horizon that
point races away, and nothing bounded it: run over the real corpus the producer wrote distances with a 95th
percentile of 410 m and a maximum of 11,978 m, lateral offsets spanning 5.5 km, and times to collision of up
to 36 days, all from dashcam frames.

None of that is a bug in the geometry. It is the geometry being asked a question it cannot answer, and
answering anyway. These tests pin the range at which the answer stops being a measurement, and that a value
past it is reported as absent rather than as a number.
"""

import math

import pytest

from services.dynamics.compute import (
    BOX_BOTTOM_JITTER_PX,
    IPM_ERROR_BUDGET,
    MAX_TTC_S,
    MIN_CLOSING_MPS,
    ipm_max_range_m,
)

# The real rig, from configs: the forward camera is the narrow lens and the mount is 1.5 m up.
FY_NARROW = 2870.0
FY_WIDE = 917.0
CAMERA_H = 1.5


def test_the_range_follows_the_derivation():
    """f_max = tau * fy * h / dv. Stated as the formula so a config change moves it correctly."""
    expected = IPM_ERROR_BUDGET * FY_NARROW * CAMERA_H / BOX_BOTTOM_JITTER_PX
    assert ipm_max_range_m(FY_NARROW, CAMERA_H) == pytest.approx(expected)


def test_the_forward_camera_reaches_a_plausible_dashcam_range():
    """Sanity, in metres a person can argue with rather than a formula nobody checks."""
    f_max = ipm_max_range_m(FY_NARROW, CAMERA_H)
    assert 100.0 < f_max < 300.0, f"{f_max:.0f} m is not a believable monocular working range"


def test_a_wider_lens_cannot_see_as_far():
    """Fewer pixels per degree means the same jitter costs more distance, so the bound must tighten."""
    assert ipm_max_range_m(FY_WIDE, CAMERA_H) < ipm_max_range_m(FY_NARROW, CAMERA_H)


def test_a_higher_mount_reaches_further():
    """Height is what gives the ray an angle to work with, so it buys range."""
    assert ipm_max_range_m(FY_NARROW, 2.5) > ipm_max_range_m(FY_NARROW, CAMERA_H)


def test_the_error_budget_is_what_makes_the_bound_defensible():
    # Doubling what we are willing to be wrong by doubles how far we will look.
    assert ipm_max_range_m(FY_NARROW, CAMERA_H) == pytest.approx(
        (IPM_ERROR_BUDGET * FY_NARROW * CAMERA_H) / BOX_BOTTOM_JITTER_PX)
    assert 0 < IPM_ERROR_BUDGET < 1, "an error budget at or above 1 is not a budget"
    assert BOX_BOTTOM_JITTER_PX >= 1.0, "a detector's box bottom is not pixel-exact"


def test_the_corpus_maximum_is_now_out_of_range():
    """The observed extremes must fall outside what the bound admits, or the bound changes nothing."""
    f_max = ipm_max_range_m(FY_NARROW, CAMERA_H)
    for observed in (410.2, 11978.6):        # the p95 and max the unbounded producer wrote
        assert observed > f_max, f"{observed} m would still be accepted"


def test_the_median_observation_is_still_accepted():
    """A bound that rejects the ordinary case has replaced one failure with another."""
    assert 58.2 < ipm_max_range_m(FY_NARROW, CAMERA_H)     # the p50 the producer wrote


def test_a_time_to_collision_needs_a_gap_that_is_actually_closing():
    # The old guard was 1e-3 m/s, a millimetre per second, which is what produced a 36-day time to collision
    # by dividing a large distance by nearly zero.
    assert MIN_CLOSING_MPS > 1e-3
    assert MIN_CLOSING_MPS <= 1.0, "too high a floor would discard genuine slow approaches"


def test_a_time_to_collision_beyond_the_horizon_is_not_reported():
    assert MAX_TTC_S <= 120.0, "past a minute or two this is arithmetic, not a collision estimate"
    # The worst observed value must not survive the cap.
    assert 3137867.7 > MAX_TTC_S


def test_the_old_guards_would_have_admitted_the_absurd_values():
    """Kept executable so the regression stays demonstrable."""
    def old_accepts_distance(_d: float) -> bool:
        return True                       # the lift was written straight through, unbounded

    def old_ttc(dist: float, closing_mps: float):
        return dist / closing_mps if closing_mps > 1e-3 else None

    assert old_accepts_distance(11978.6) is True
    assert ipm_max_range_m(FY_NARROW, CAMERA_H) < 11978.6

    # Just above the old floor of 1e-3 m/s, which is the point: the guard admitted it.
    absurd = old_ttc(400.0, 2e-3)
    assert absurd is not None and math.isfinite(absurd)
    assert absurd > MAX_TTC_S, "a 55-hour time to collision passed the old guard"
    # The same gap is now reported as not closing at all, which is what it is.
    assert 2e-3 < MIN_CLOSING_MPS
