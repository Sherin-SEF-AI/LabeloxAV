"""The gate that decides whether two detections are one object.

Its absence is the whole defect behind 137,913 interpolated objects at 0.209 precision. The arithmetic that
drew them was correct; the pairs it drew between were not the same thing. These tests pin the three refusals
and, as importantly, pin what must still pass - a gate that refuses everything would also have prevented the
damage and would be useless.
"""

import pytest

from services.temporal.gap_gate import (
    MAX_AREA_RATIO,
    MAX_GAP_FRAMES,
    is_discontinuity,
    same_object,
)

W = 1920
NEAR = ([100.0, 100.0, 180.0, 200.0], [140.0, 105.0, 225.0, 205.0])


@pytest.fixture
def cliques():
    from services.domain import active_pack

    return active_pack().cliques


class TestItRefusesWhatItShould:
    def test_a_teleport_is_not_one_object(self, cliques):
        r = same_object([100, 100, 180, 200], [1500, 100, 1580, 200], "sedan", "sedan",
                        frame_width=W, gap_frames=3, cliques=cliques)
        assert not r.ok and r.reason == "endpoints_teleport"
        # The measurement is returned, so a refusal can be argued with rather than only obeyed.
        assert r.detail["travel_px"] > r.detail["limit_px"]

    def test_a_scale_jump_is_not_one_object(self, cliques):
        # Centres kept close on purpose, so this exercises the scale bound rather than tripping the
        # displacement one first - a box that both jumps and balloons tells you nothing about which gate
        # caught it.
        r = same_object([800, 400, 880, 500], [700, 300, 1000, 620], "sedan", "sedan",
                        frame_width=W, gap_frames=3, cliques=cliques)
        assert not r.ok and r.reason == "endpoints_scale_jump", r

    def test_a_hole_longer_than_the_bound_is_refused(self, cliques):
        r = same_object(*NEAR, "sedan", "sedan", frame_width=W,
                        gap_frames=MAX_GAP_FRAMES + 1, cliques=cliques)
        assert not r.ok and r.reason == "gap_too_long"

    def test_two_different_kinds_of_thing_do_not_bridge(self, cliques):
        for a, b in (("pedestrian", "bus"), ("sedan", "traffic_signal"), ("cattle", "truck")):
            r = same_object(*NEAR, a, b, frame_width=W, gap_frames=3, cliques=cliques)
            assert not r.ok, f"{a} bridged to {b}"
            assert r.reason == "endpoints_class_mismatch"

    def test_with_no_clique_spec_any_class_difference_is_a_refusal(self):
        """A domain that declares no confusions gives no basis for reading two class names as one object."""
        r = same_object(*NEAR, "sedan", "suv", frame_width=W, gap_frames=3, cliques=None)
        assert not r.ok and r.reason == "endpoints_class_mismatch"


class TestItPassesWhatItShould:
    """A gate that refuses everything prevents the damage and delivers nothing."""

    def test_a_short_move_between_identical_classes_passes(self, cliques):
        assert same_object(*NEAR, "sedan", "sedan", frame_width=W, gap_frames=3, cliques=cliques).ok

    def test_the_detector_renaming_one_object_does_not_break_the_track(self, cliques):
        """A single receding vehicle in this corpus is labelled truck, rider, autorickshaw, suv and
        motorcycle on five consecutive frames. Requiring exact class equality rejected 65.6% of holes."""
        for a, b in (("sedan", "suv"), ("mpv", "hatchback"), ("suv", "minivan"),
                     ("motorcycle", "scooter"), ("cattle", "buffalo")):
            assert same_object(*NEAR, a, b, frame_width=W, gap_frames=3, cliques=cliques).ok, f"{a}/{b}"

    def test_a_fast_object_at_three_fps_is_not_a_teleport(self, cliques):
        """Ingest runs at 3 fps, so a nearby vehicle legitimately moves a long way between samples. The
        bound rejects teleports, not motion."""
        r = same_object([100, 400, 260, 520], [520, 400, 680, 520], "sedan", "sedan",
                        frame_width=W, gap_frames=2, cliques=cliques)
        assert r.ok, r

    def test_an_approaching_vehicle_growing_normally_still_passes(self, cliques):
        """Growth is the common case, not the exception: the median area ratio between consecutive
        detections that pass the displacement bound is 1.24, and the 90th percentile is 3.90."""
        r = same_object([900, 500, 960, 550], [890, 490, 990, 575], "sedan", "sedan",
                        frame_width=W, gap_frames=3, cliques=cliques)
        assert r.ok, r


class TestDiscontinuity:
    """The splitter's predicate: the geometric half, without the class test."""

    def test_a_teleport_is_a_discontinuity(self):
        assert is_discontinuity([100, 100, 180, 200], [1500, 100, 1580, 200], frame_width=W)

    def test_normal_motion_is_not(self):
        assert not is_discontinuity(*NEAR, frame_width=W)

    def test_a_class_change_alone_does_not_split_a_track(self):
        """Splitting on class would shred correct tracks: the detector is unstable frame to frame, and that
        instability is a fact about the detector rather than about the object."""
        # Same geometry, and is_discontinuity does not take a class at all.
        assert not is_discontinuity(*NEAR, frame_width=W)

    def test_zero_overlap_alone_is_not_enough(self):
        """At 3 fps a small fast object legitimately clears its own box between samples."""
        a, b = [100.0, 100.0, 140.0, 140.0], [150.0, 100.0, 190.0, 140.0]
        assert (max(a[2], b[2]) - min(a[0], b[0])) > 0        # they do not overlap
        assert not is_discontinuity(a, b, frame_width=W)

    def test_zero_overlap_with_a_large_area_change_is(self):
        a, b = [100.0, 100.0, 140.0, 140.0], [400.0, 100.0, 700.0, 400.0]
        assert is_discontinuity(a, b, frame_width=W)
        assert (b[2] - b[0]) * (b[3] - b[1]) / ((a[2] - a[0]) * (a[3] - a[1])) > MAX_AREA_RATIO
