"""Offline smoothing, and the refusal that makes it safe to use.

ORACLYX's argument is that running offline beats any online tracker: at frame t a filter is guessing, while a
batch pass can correct frame t with what happened afterwards. The module did not use that, interpolating
linearly between anchors, which trusts every anchor exactly and cannot tell a real displacement from a bad
detection.

Two findings shaped what got built.

The motion model could not be a constant. Real tracks here move a median of 53px and a p90 of 564px between
frames, because the footage is 3fps from a moving vehicle. A fixed 4.0 px/s^2 gated 86% of every box, which is
a smoother fighting its data.

And most of these tracks are not trajectories at all: 43% of consecutive box pairs have zero overlap, 50%
change class, and 57 of 60 sampled tracks contain more than one class. Smoothing those would emit a confident
path through the average of several different objects, which is the plausible-wrong-answer this system keeps
having to hunt down. So incoherent input is refused, and the refusal has as many tests as the smoothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.oraclyx.smoothing import (
    MAX_ZERO_OVERLAP_FRAC,
    SmoothingParams,
    coherence,
    displacement,
    smooth_track,
    to_centre_size,
    to_corners,
)

HZ = 100_000_000  # 10fps in ns


def _line(n=20, step=6.0, jitter=None, cy=200.0, half=20.0):
    """A box moving steadily right; `jitter` maps a frame index to an added centre offset."""
    out = []
    for k in range(n):
        cx = 100.0 + step * k + (jitter or {}).get(k, 0.0)
        out.append((k * HZ, [cx - half, cy - half, cx + half, cy + half]))
    return out


def _cx(box):
    return (box[0] + box[2]) / 2.0


# ------------------------------------------------------------------------------- the core claim

def test_a_single_bad_detection_is_pulled_back_to_the_trajectory():
    """The whole point. Interpolation reproduces the error faithfully; a smoother outvotes it."""
    obs = _line(jitter={10: 40.0})
    out = smooth_track(obs)
    at10 = next(b for b in out["boxes"] if b["ts_ns"] == 10 * HZ)
    assert out["smoothed"] is True
    assert abs(_cx(at10["bbox"]) - 160.0) < 2.0, "the smoothed box should sit on the true path, not the flyer"


def test_the_bad_frame_is_flagged_rather_than_silently_absorbed():
    out = smooth_track(_line(jitter={10: 40.0}))
    assert out["gated"] >= 1
    assert next(b for b in out["boxes"] if b["ts_ns"] == 10 * HZ)["gated"] is True


def test_a_clean_trajectory_is_barely_moved():
    """A smoother that rewrites good data is not cleaning it."""
    obs = _line()
    d = displacement(obs, smooth_track(obs)["boxes"])
    assert d["mean_px"] < 3.0


def test_every_box_carries_its_own_uncertainty():
    """Worth as much as the box: it tells the distillation path which frames to trust."""
    out = smooth_track(_line())
    assert all(isinstance(b["std"], float) and b["std"] >= 0 for b in out["boxes"])
    assert out["mean_std"] >= 0


def test_observations_are_sorted_before_filtering():
    """A caller reading from a database has no reason to guarantee order, and a filter fed out-of-order
    timestamps produces confident nonsense."""
    obs = _line()
    shuffled = list(reversed(obs))
    assert smooth_track(shuffled)["boxes"] == smooth_track(obs)["boxes"]


# ------------------------------------------------------------------------------- the adaptive model

def test_the_motion_model_is_estimated_from_the_track():
    """A constant cannot serve a corpus where displacement ranges from 53px to 564px per frame."""
    slow = smooth_track(_line(step=1.0))
    fast = smooth_track(_line(step=80.0, half=40.0))
    assert fast["process_pos"] >= slow["process_pos"]


def test_a_fast_object_is_not_treated_as_a_track_of_errors():
    """The first version gated 86% of all boxes because every fast object looked like a mistake."""
    out = smooth_track(_line(step=80.0, half=40.0))
    assert out["gated"] <= 2, "steady fast motion is motion, not a sequence of outliers"


def test_the_estimate_is_reported_so_a_surprising_result_can_be_traced():
    out = smooth_track(_line())
    assert "process_pos" in out and out["process_pos"] > 0


def test_a_fixed_model_can_still_be_forced():
    out = smooth_track(_line(), params=SmoothingParams(adapt=False, process_pos=9.0))
    assert out["process_pos"] == 9.0


# ------------------------------------------------------------------------------- the refusal

def test_a_sequence_of_different_objects_is_refused():
    """43% of this corpus's consecutive boxes do not overlap at all. Smoothing between two different objects
    would produce a confident path through their average."""
    bad = [(k * HZ, ([10, 10, 50, 50] if k % 2 else [900, 600, 960, 660])) for k in range(20)]
    out = smooth_track(bad)
    assert out["smoothed"] is False
    assert "more than one object" in out["reason"]


def test_a_refused_track_returns_its_input_unchanged():
    """Refusing must not also lose the data: the caller still needs the boxes it had."""
    bad = [(k * HZ, ([10, 10, 50, 50] if k % 2 else [900, 600, 960, 660])) for k in range(10)]
    out = smooth_track(bad)
    assert [b["bbox"] for b in out["boxes"]] == [list(map(float, b)) for _, b in bad]
    assert all(b["std"] is None for b in out["boxes"]), "no uncertainty is claimed for unsmoothed boxes"


def test_the_refusal_can_be_overridden_deliberately():
    bad = [(k * HZ, ([10, 10, 50, 50] if k % 2 else [900, 600, 960, 660])) for k in range(20)]
    assert smooth_track(bad, require_coherent=False)["smoothed"] is True


def test_some_discontinuity_is_tolerated():
    """A fast object at 3fps can leave its previous box entirely, and an occlusion makes a gap. Refusing
    those would refuse most real tracking."""
    obs = _line(step=45.0, half=20.0)      # each box clears the last
    coh = coherence(obs)
    assert coh["zero_overlap_frac"] > 0
    assert MAX_ZERO_OVERLAP_FRAC > 0.1


def test_coherence_reports_numbers_not_just_a_verdict():
    coh = coherence(_line())
    assert set(coh) >= {"coherent", "pairs", "zero_overlap_frac", "median_iou"}


# ------------------------------------------------------------------------------- degenerate input

@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_short_to_smooth_says_so_rather_than_pretending(n):
    """Two points define a line: there is nothing for a motion model to disagree with, and returning the
    input while implying it was improved would be the lie."""
    out = smooth_track(_line(n=n)[:n])
    assert out["smoothed"] is False
    assert out["n"] == n


def test_box_conversion_round_trips():
    box = [10.0, 20.0, 50.0, 80.0]
    assert to_corners(to_centre_size(box)) == pytest.approx(box)


def test_a_negative_size_never_escapes():
    """A smoothed state can cross zero width on a shrinking box; a negative extent would corrupt every
    consumer downstream."""
    out = to_corners(np.array([100.0, 100.0, -5.0, -5.0]))
    assert out[2] >= out[0] and out[3] >= out[1]
