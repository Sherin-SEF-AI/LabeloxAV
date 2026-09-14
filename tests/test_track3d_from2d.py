"""Track-level 3D: one frame, one size, one smoothed trajectory, and the refusals in between.

The frame transform is the part that can be silently wrong: a sign error in the yaw produces a
trajectory that is mirrored about the driving direction and looks completely plausible on a plot. So the
transform is pinned against hand-computed cases rather than against itself.
"""

from __future__ import annotations

import math

import pytest

from services.lidar.track3d.from2d import (
    MIN_CUBOIDS_FOR_MEDIAN,
    dimension_variance,
    ego_to_world,
    lock_dimensions,
    smooth_positions,
)


class Pose:
    """The fields `ego_to_world` reads off an EgoPose row."""

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, measured=False, source="visual"):
        self.x, self.y, self.z = x, y, z
        self.qw, self.qz = math.cos(yaw / 2), math.sin(yaw / 2)
        self.measured, self.source = measured, source


class TestTheFrameTransform:
    def test_at_the_origin_facing_east_the_frames_agree(self):
        assert ego_to_world([10.0, 0.0, 1.0], Pose()) == pytest.approx([10.0, 0.0, 1.0])

    def test_the_pose_translates_the_box(self):
        assert ego_to_world([10.0, 0.0, 0.0], Pose(x=5.0, y=3.0)) == pytest.approx([15.0, 3.0, 0.0])

    def test_a_quarter_turn_puts_forward_to_the_north(self):
        got = ego_to_world([10.0, 0.0, 0.0], Pose(yaw=math.pi / 2))
        assert got[0] == pytest.approx(0.0, abs=1e-9)
        assert got[1] == pytest.approx(10.0)

    def test_left_of_the_vehicle_is_north_when_facing_east(self):
        """Ego frame is x forward, y left. A sign error here mirrors every trajectory about the
        driving direction and still plots as a plausible road."""
        got = ego_to_world([0.0, 4.0, 0.0], Pose())
        assert got[1] == pytest.approx(4.0)

    def test_height_passes_through_and_adds(self):
        assert ego_to_world([0.0, 0.0, 1.6], Pose(z=0.4))[2] == pytest.approx(2.0)

    def test_two_boxes_seen_from_two_poses_land_on_the_same_point(self):
        """The property the whole table exists for: the same real object, lifted in two different ego
        frames, is one point in the session frame."""
        a = ego_to_world([20.0, 0.0, 0.0], Pose(x=0.0, y=0.0))
        b = ego_to_world([10.0, 0.0, 0.0], Pose(x=10.0, y=0.0))
        assert a == pytest.approx(b)


class TestLockedDimensions:
    def test_the_median_survives_one_bad_frame(self):
        dims = [[4.0, 1.8, 1.5], [4.1, 1.8, 1.5], [40.0, 18.0, 15.0], [4.0, 1.8, 1.5]]
        assert lock_dimensions(dims) == pytest.approx([4.05, 1.8, 1.5])

    def test_too_few_cuboids_refuses_rather_than_locking_to_noise(self):
        assert lock_dimensions([[4.0, 1.8, 1.5]] * (MIN_CUBOIDS_FOR_MEDIAN - 1)) is None

    def test_variance_is_what_locking_removes(self):
        dims = [[4.0, 1.8, 1.5], [4.4, 1.9, 1.6], [3.6, 1.7, 1.4]]
        before = dimension_variance(dims)
        assert before[0] > 0
        locked = lock_dimensions(dims)
        assert dimension_variance([locked] * 3) == pytest.approx([0.0, 0.0, 0.0])


class TestSmoothing:
    def test_a_straight_run_stays_straight(self):
        pts = [(i * 100_000_000, [float(i), 0.0, 0.0]) for i in range(10)]
        out = smooth_positions(pts)
        assert out[-1][0] == pytest.approx(9.0, abs=0.5)
        assert all(abs(p[1]) < 0.2 for p in out)

    def test_a_single_spike_is_pulled_back_toward_the_line(self):
        pts = [(i * 100_000_000, [float(i), 0.0, 0.0]) for i in range(10)]
        pts[5] = (pts[5][0], [5.0, 8.0, 0.0])
        out = smooth_positions(pts)
        assert abs(out[5][1]) < 8.0, "the outlier moved toward the trajectory"

    def test_too_few_points_are_returned_unchanged(self):
        pts = [(0, [1.0, 2.0, 3.0]), (1, [4.0, 5.0, 6.0])]
        assert smooth_positions(pts) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_uneven_timestamps_are_not_treated_as_even(self):
        """Dashcam frames are not evenly spaced. Treating a two-second gap as one frame step turns it
        into acceleration that never happened."""
        even = smooth_positions([(i * 100_000_000, [float(i), 0.0, 0.0]) for i in range(6)])
        gapped = smooth_positions([(0, [0.0, 0.0, 0.0]), (100_000_000, [1.0, 0.0, 0.0]),
                                   (2_000_000_000, [2.0, 0.0, 0.0]), (2_100_000_000, [3.0, 0.0, 0.0]),
                                   (2_200_000_000, [4.0, 0.0, 0.0]), (2_300_000_000, [5.0, 0.0, 0.0])])
        assert even != gapped
