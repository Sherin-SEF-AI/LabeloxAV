"""Lane polylines from a marking mask: two vertical marking lines yield two ordered polylines that track
their x positions; specks and too-short marks are filtered."""

from __future__ import annotations

import numpy as np

from services.autolabel.lane.marking import lanes_from_marking_mask


def _mask_with_two_lines():
    m = np.zeros((400, 600), np.uint8)
    m[50:350, 150:156] = 1          # left line at x~152
    m[50:350, 440:446] = 1          # right line at x~442
    return m


def test_two_lines_become_two_polylines():
    lanes = lanes_from_marking_mask(_mask_with_two_lines())
    assert len(lanes) == 2
    xs = sorted(np.mean([p[0] for p in lane]) for lane in lanes)
    assert abs(xs[0] - 152) < 4 and abs(xs[1] - 442) < 4     # polylines track the marking x positions


def test_polyline_spans_the_line_vertically():
    lane = lanes_from_marking_mask(_mask_with_two_lines())[0]
    assert lane[0][1] < 80 and lane[-1][1] > 320            # samples from top to bottom of the line


def test_specks_and_short_marks_filtered():
    m = np.zeros((400, 600), np.uint8)
    m[10:14, 10:14] = 1             # tiny speck (too few pixels, too short)
    m[100:108, 300:360] = 1         # short horizontal mark (height < min_height)
    assert lanes_from_marking_mask(m) == []


def test_empty_mask():
    assert lanes_from_marking_mask(np.zeros((400, 600), np.uint8)) == []


# ---- dashes are one lane, not one lane per dash -------------------------------------------------------

def _dashed_mask(on: int, off: int, x: int = 300, h: int = 400, w: int = 600) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    y = 50
    while y < h - 50:
        m[y:min(y + on, h - 50), x:x + 6] = 1
        y += on + off
    return m


def test_a_dashed_lane_is_one_lane_and_not_one_lane_per_dash():
    """Connected components cannot represent a dashed line: each dash is its own component. Storing them
    separately is not merely untidy now that lane type is measured, because a stub is short enough to be
    entirely paint and therefore reads as a confident solid line, and crossing a solid line is an offence.
    Fragmenting a dashed lane manufactures violations."""
    from services.autolabel.lane.marking import lanes_from_marking_mask_grouped

    mask = _dashed_mask(on=30, off=30)
    assert len(lanes_from_marking_mask(mask)) == 5, "the ungrouped path sees five lanes"
    grouped = lanes_from_marking_mask_grouped(mask, frame_width=600)
    assert len(grouped) == 1
    assert grouped[0][-1][1] - grouped[0][0][1] > 200, "and it spans the whole run of dashes"


def test_short_dashes_survive_instead_of_vanishing():
    """At the old whole-lane minimums a lane with short dashes came back as nothing at all, so the road
    simply had no lane there."""
    from services.autolabel.lane.marking import lanes_from_marking_mask_grouped

    mask = _dashed_mask(on=14, off=20)
    assert lanes_from_marking_mask(mask) == [], "the old thresholds discarded every dash"
    assert len(lanes_from_marking_mask_grouped(mask, frame_width=600)) == 1


def test_two_dashed_lanes_stay_two_lanes():
    """Grouping must not be so eager that neighbouring lanes collapse into one."""
    from services.autolabel.lane.marking import lanes_from_marking_mask_grouped

    mask = np.maximum(_dashed_mask(30, 30, x=200), _dashed_mask(30, 30, x=420))
    assert len(lanes_from_marking_mask_grouped(mask, frame_width=600)) == 2


def test_a_solid_line_is_unaffected_by_grouping():
    from services.autolabel.lane.marking import lanes_from_marking_mask_grouped

    m = np.zeros((400, 600), np.uint8)
    m[50:350, 300:306] = 1
    assert len(lanes_from_marking_mask_grouped(m, frame_width=600)) == 1


def test_a_stub_that_merely_points_at_a_lane_is_not_merged_into_it():
    """One-way agreement is what a short badly fitted fragment looks like when it happens to aim at a lane.
    Requiring each fit to predict the other's midpoint is what keeps it out."""
    from services.autolabel.lane.marking import group_collinear

    lane = [[300.0, 50.0], [300.0, 150.0], [300.0, 250.0]]
    # Steeply angled and far off to the side: its own line does not pass through the lane's midpoint.
    stub = [[380.0, 300.0], [368.0, 320.0]]
    assert len(group_collinear([lane, stub], frame_width=600)) == 2


def test_fragments_that_cannot_be_fitted_are_carried_rather_than_dropped():
    from services.autolabel.lane.marking import group_collinear

    lane = [[300.0, 50.0], [300.0, 250.0]]
    degenerate = [[10.0, 10.0]]
    out = group_collinear([lane, degenerate], frame_width=600)
    assert len(out) == 2, "a fragment nobody can fit is still marking pixels somebody found"
