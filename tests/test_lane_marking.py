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
