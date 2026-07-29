"""Whether a proposed lane is on the road at all.

The interesting cases are the refusals to refuse. A road-edge lane runs along the boundary of the drivable
region and its points sit slightly outside it, so a naive containment test deletes exactly the lanes the
ontology cares most about. A frame nobody has segmented has no surface to check against, and refusing its
lanes would make the proposer depend on a second model having run first.
"""

from __future__ import annotations

from services.autolabel.lane.plausible import (
    MIN_ON_SURFACE,
    filter_proposals,
    is_plausible,
    on_surface_fraction,
    surface_polygons,
)

W, H = 1920, 1080

# A road occupying the lower middle of the frame, as a flattened polygon the way the drivable mask stores it.
ROAD = [400.0, 700.0, 1500.0, 700.0, 1700.0, 1070.0, 200.0, 1070.0]
CLASSES = {"drivable": [ROAD], "non_drivable": [], "fallback": []}


def _lane(pts):
    return [[float(x), float(y)] for x, y in pts]


def test_a_lane_down_the_middle_of_the_road_is_kept():
    lane = _lane([(800, 750), (830, 850), (860, 950), (890, 1050)])
    ok, ev = is_plausible(lane, CLASSES, W)
    assert ok is True
    assert ev["on_surface"] == 1.0


def test_a_line_across_the_sky_is_rejected():
    """The case this exists for. Six of these on one real frame, every one an edge of a striped hoarding,
    drawn as diagonals over the sky and obliterating a correct drivable overlay underneath."""
    # Genuinely above the road, which starts at y=700. The first version of this fixture ran diagonally
    # *through* the road polygon and scored 0.75, which is the test being wrong rather than the rule.
    lane = _lane([(200, 150), (700, 260), (1200, 350), (1700, 430)])
    ok, ev = is_plausible(lane, CLASSES, W)
    assert ok is False
    assert ev["on_surface"] < MIN_ON_SURFACE
    assert "not lie on the road" in ev["reason"]


def test_a_road_edge_lane_just_outside_the_surface_is_kept():
    """A lane boundary runs along the edge of the road, so its points legitimately fall a little outside it.
    Strict containment would delete every road edge, which is the opposite of the intent."""
    # Consistently 30px left of the left boundary, which the slant makes about 26px perpendicular: outside
    # the polygon, inside the 57px margin.
    lane = _lane([(343, 750), (289, 850), (235, 950), (181, 1050)])
    strict = on_surface_fraction(lane, [ROAD], margin_px=0.0)
    assert strict < 0.5, "strictly outside the polygon"
    ok, ev = is_plausible(lane, CLASSES, W)
    assert ok is True, "but inside the margin, so it is still a lane"
    assert ev["margin_px"] > 20


def test_a_frame_with_no_drivable_mask_keeps_its_lanes():
    """Missing evidence, not evidence of absence. A proposer that produced nothing until somebody had run
    the segmenter would be a worse failure than the one this fixes."""
    ok, ev = is_plausible(_lane([(0, 0), (10, 10)]), None, W)
    assert ok is True
    assert ev["checked"] is False

    ok2, ev2 = is_plausible(_lane([(0, 0), (10, 10)]), {"drivable": []}, W)
    assert ok2 is True
    assert ev2["checked"] is False


def test_fallback_surface_counts_as_road_and_pavement_does_not():
    """Fallback is the unpaved shoulder India drives on, so a lane there is real. Non-drivable is the
    pavement and the median, and a lane line does not run along those."""
    assert surface_polygons({"drivable": [ROAD]}) == [ROAD]
    assert surface_polygons({"fallback": [ROAD]}) == [ROAD]
    assert surface_polygons({"non_drivable": [ROAD]}) == []

    lane = _lane([(800, 750), (830, 850), (860, 950)])
    assert is_plausible(lane, {"fallback": [ROAD]}, W)[0] is True
    assert is_plausible(lane, {"non_drivable": [ROAD]}, W)[0] is True, \
        "no road polygons at all means nothing to check, so it is kept"


def test_a_road_split_by_an_island_is_one_surface():
    """The road is often several disjoint polygons either side of a traffic island, and a lane on the far
    side must not be rejected for being outside the near one."""
    left = [200.0, 700.0, 800.0, 700.0, 800.0, 1070.0, 200.0, 1070.0]
    right = [1100.0, 700.0, 1700.0, 700.0, 1700.0, 1070.0, 1100.0, 1070.0]
    lane = _lane([(1300, 750), (1350, 900), (1400, 1050)])
    ok, _ev = is_plausible(lane, {"drivable": [left, right]}, W)
    assert ok is True


def test_filtering_returns_both_halves_with_their_evidence():
    """A proposer that silently halves its own output is one nobody can debug."""
    good = _lane([(800, 750), (830, 850), (860, 950)])
    bad = _lane([(50, 100), (100, 150), (150, 200)])
    kept, rejected = filter_proposals([good, bad], CLASSES, W)
    assert len(kept) == 1 and len(rejected) == 1
    assert kept[0][0] == good
    assert rejected[0][1]["on_surface"] == 0.0
    assert rejected[0][1]["checked"] is True


def test_a_degenerate_polygon_is_not_a_surface():
    """Two points is a line, not a region, and testing containment against it would reject everything."""
    assert on_surface_fraction(_lane([(800, 800)]), [[0.0, 0.0, 10.0, 10.0]], 50.0) is None
    assert is_plausible(_lane([(800, 800)]), {"drivable": [[0.0, 0.0, 10.0, 10.0]]}, W)[0] is True


def test_an_empty_lane_is_not_scored():
    assert on_surface_fraction([], [ROAD], 50.0) is None
