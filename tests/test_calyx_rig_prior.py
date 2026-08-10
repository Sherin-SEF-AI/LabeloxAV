"""A fleet prior that says which of its axes it actually measured.

CALYX's fusion worked and had never been run on the corpus. Running it first is what made this worth
building, because the corpus does not look like the code assumes.

All 101 calibrations are source='estimated' with quality=0.6, ninety-seven of them on one vehicle, and every
single one carries fx=fy=2870, cx=960, cy=540, xyz=[0, 0, 1.5], roll=0, yaw=0. Only pitch differs. Five of
the six extrinsic degrees of freedom are a constant that was written once and copied.

`fuse_calibrations` pools all of rpy into one spread, so on this input it reports a spread near zero and a
high confidence, which reads as ninety-seven sessions independently agreeing to six decimal places. It is the
strongest claim in the module resting on the weakest evidence in the corpus, and it is the same shape as the
gold set that named 400 objects and resolved 47.

So these tests are mostly about the refusal: a degenerate axis gets no sigma, cannot be an outlier, and
visibly discounts the prior's confidence.
"""

from __future__ import annotations

import pytest

from services.calyx.rig_prior import (
    OUTLIER_SIGMA,
    build_rig_prior,
    deviations,
    prior_confidence,
)


def _calib(pitch: float, *, roll: float = 0.0, yaw: float = 0.0, z: float = 1.5) -> dict:
    """The corpus shape: everything fixed except pitch."""
    return {"rpy_deg": [roll, pitch, yaw], "xyz_m": [0.0, 0.0, z]}


# A pitch spread resembling DASHCAM-01's: a cluster around half a degree.
FLEET = [_calib(p) for p in (0.50, 0.54, 0.48, 0.60, 0.51, 0.55, 0.47, 0.53, 0.58, 0.49)]


# ------------------------------------------------------------------------------- what was measured

def test_the_prior_separates_measured_axes_from_copied_ones():
    """The whole point. One pooled spread over rpy cannot express this and the corpus is entirely this."""
    p = build_rig_prior(FLEET)
    assert p["measured_axes"] == ["pitch"]
    assert set(p["constant_axes"]) == {"roll", "yaw", "x", "y", "z"}


def test_a_constant_axis_is_given_no_sigma_at_all():
    """0.0 would be a tolerance nobody measured, and dividing by it makes every session infinitely deviant."""
    p = build_rig_prior(FLEET)
    assert p["axes"]["roll"]["sigma"] is None
    assert p["axes"]["pitch"]["sigma"] is not None and p["axes"]["pitch"]["sigma"] > 0


def test_the_finding_is_stated_in_words_not_only_in_numbers():
    """A caller reading only the numbers would treat five defaults as five agreements."""
    d = build_rig_prior(FLEET)["detail"]
    assert "pitch" in d and "carry no evidence" in d


def test_an_axis_that_varies_everywhere_is_measured_everywhere():
    varied = [{"rpy_deg": [r, p, y], "xyz_m": [x, 0.1 * i, 1.5 + 0.01 * i]}
              for i, (r, p, y, x) in enumerate([(0.1, 0.5, 0.2, 0.0), (0.2, 0.6, 0.3, 0.1),
                                                (0.15, 0.55, 0.25, 0.05), (0.3, 0.7, 0.4, 0.2)])]
    p = build_rig_prior(varied)
    assert "roll" in p["measured_axes"] and "yaw" in p["measured_axes"]


def test_the_scale_is_robust_to_a_badly_calibrated_session():
    """A standard deviation is set by the outlier it is meant to find, which is how a drifted rig hides."""
    clean = build_rig_prior(FLEET)["axes"]["pitch"]["sigma"]
    withbad = build_rig_prior([*FLEET, _calib(45.0)])["axes"]["pitch"]["sigma"]
    assert withbad < clean * 3, "one wild session must not triple the tolerance"


# ------------------------------------------------------------------------------- deviations

def test_a_drifted_session_is_flagged_on_the_measured_axis():
    p = build_rig_prior(FLEET)
    d = deviations(_calib(3.0), p)
    assert d["outlier"] is True
    assert d["flagged_axes"] == ["pitch"]
    assert d["axes"]["pitch"]["sigmas"] >= OUTLIER_SIGMA


def test_a_normal_session_is_not_flagged():
    p = build_rig_prior(FLEET)
    d = deviations(_calib(0.52), p)
    assert d["outlier"] is False
    assert "within" in d["detail"]


def test_a_difference_on_a_constant_axis_is_reported_but_never_called_an_outlier():
    """With no observed variation there is no scale, so a difference is different, not anomalous. Scoring it
    would be arithmetic dressed as evidence, and on this corpus it would fire on every axis at once."""
    p = build_rig_prior(FLEET)
    d = deviations({"rpy_deg": [9.0, 0.52, 0.0], "xyz_m": [0.0, 0.0, 1.5]}, p)
    roll = d["axes"]["roll"]
    assert roll["delta"] == pytest.approx(9.0)
    assert roll["sigmas"] is None
    assert roll["outlier"] is False
    assert "cannot be scored" in roll["note"]
    assert d["outlier"] is False


def test_deviation_against_an_empty_prior_says_so_rather_than_dividing():
    assert deviations(_calib(0.5), build_rig_prior([]))["outlier"] is False


# ------------------------------------------------------------------------------- confidence

def test_confidence_is_discounted_by_how_little_was_measured():
    """Ninety-seven sessions that only ever measured pitch must not read like ninety-seven full calibrations."""
    corpus_like = build_rig_prior([_calib(0.5 + 0.01 * i) for i in range(97)])
    everything = build_rig_prior([
        {"rpy_deg": [0.1 * i, 0.5 + 0.01 * i, 0.2 * i], "xyz_m": [0.01 * i, 0.02 * i, 1.5 + 0.01 * i]}
        for i in range(97)])
    assert prior_confidence(corpus_like) < prior_confidence(everything)
    assert prior_confidence(corpus_like) <= 1 / 6 + 1e-6, "one measured axis of six caps the claim"


def test_no_calibrations_is_no_confidence():
    assert prior_confidence(build_rig_prior([])) == 0.0


def test_an_empty_fleet_does_not_raise():
    p = build_rig_prior([])
    assert p["n"] == 0 and p["axes"] == {}
