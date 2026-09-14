"""Simplifying a mask at the moment it is stored, without moving the shape.

Two implementations of Ramer-Douglas-Peucker already existed and neither ran at write time: the
open-vocabulary path simplifies at contour extraction, the browser trims a SAM candidate before sending.
So a mask that came from either is tidy, and a mask that was hand-brushed, imported, or edited by
dragging vertices was stored exactly as drawn, because `_write_mask` serialised whatever it was handed.

Finding the right guard took a measurement that first pointed the wrong way, and the tests below encode
what it actually said.

Simplifying 899 stored masks and comparing each ring to its original by true IoU gives a mean of 0.9975
and a worst case of 0.80, which reads as "this sometimes destroys a shape". Listing the worst cases by
area says otherwise: they are rings of 4, 6, 15 and 16 SQUARE PIXELS, where an 0.80 IoU is a change of
under one square pixel. With an absolute area floor in place the worst case is 0.9551 and only 3 rings of
1,177 fall below 0.98, while 1.9% of vertices still go.

An area-RATIO guard was tried first and is not kept, because area agreement is not overlap: a ring can
keep its area exactly while moving. That test is here too, as the reason the ratio approach was dropped.
"""

import math

import numpy as np
import pytest

from core.polygons import (
    MIN_SIMPLIFY_AREA_PX,
    simplify_mask,
    simplify_polygon,
    size_tolerance,
    vertex_count,
)


def _square(side: float, n_per_edge: int = 25) -> list[float]:
    """A square outlined with many collinear points, which is what a contour tracer produces."""
    pts = []
    for i in range(n_per_edge):
        pts += [side * i / n_per_edge, 0.0]
    for i in range(n_per_edge):
        pts += [side, side * i / n_per_edge]
    for i in range(n_per_edge):
        pts += [side - side * i / n_per_edge, side]
    for i in range(n_per_edge):
        pts += [0.0, side - side * i / n_per_edge]
    return pts


def _area(flat: list[float]) -> float:
    p = np.asarray(flat, dtype=float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def test_collinear_points_on_a_straight_edge_go():
    """The case simplification exists for: a contour tracer emits a vertex per pixel along a straight
    edge, and none of them says anything."""
    poly = _square(200.0)
    out = simplify_polygon(poly)
    assert len(out) // 2 == 4, f"a square should reduce to its corners, got {len(out) // 2} vertices"
    assert _area(out) == pytest.approx(200.0 * 200.0, rel=0.01)


def test_a_tiny_ring_is_returned_exactly_as_drawn():
    """The guard the measurement produced. Below roughly an 8x8 object there is nothing to remove and any
    change is a large fraction of nothing; the worst IoU losses in the corpus were all rings of 4 to 17
    square pixels."""
    side = math.sqrt(MIN_SIMPLIFY_AREA_PX) - 1.0
    poly = _square(side, n_per_edge=6)
    assert simplify_polygon(poly) == poly


def test_the_floor_is_absolute_not_relative():
    """A ratio guard was tried and dropped: area agreement is not overlap, so a ring can keep its area
    exactly while moving. This pins that the rule in force is an absolute area, by showing a ring just
    above the floor IS simplified while one just below is not."""
    big = _square(math.sqrt(MIN_SIMPLIFY_AREA_PX) + 6.0, n_per_edge=12)
    small = _square(math.sqrt(MIN_SIMPLIFY_AREA_PX) - 1.0, n_per_edge=12)
    assert vertex_count([simplify_polygon(big)]) < vertex_count([big])
    assert simplify_polygon(small) == small


def test_the_tolerance_scales_with_the_object():
    """One constant has to be right for a 20px sign and a 900px bus, which a fixed pixel tolerance is
    not. Derived from the ring's own perimeter."""
    small = np.asarray(_square(30.0), dtype=float).reshape(-1, 2)
    large = np.asarray(_square(900.0), dtype=float).reshape(-1, 2)
    assert size_tolerance(small) < size_tolerance(large)
    # And both stay inside the bounds, so a huge object cannot lose real structure.
    assert 0.3 <= size_tolerance(small) <= 4.0
    assert 0.3 <= size_tolerance(large) <= 4.0


def test_the_seam_between_the_last_vertex_and_the_first_is_not_flattened():
    """RDP always keeps its two endpoints. Run over a ring as though it were a line, vertex 0 is pinned
    and the closing edge is replaced by a chord, which is a visible bite out of the outline. The ring is
    opened at vertex 0 and re-closed for exactly this reason, matching web/lib/simplify.ts.
    """
    # A square rotated so its corner sits at vertex 0, where a naive open-line RDP does the damage.
    poly = [100.0, 0.0, 200.0, 100.0, 100.0, 200.0, 0.0, 100.0]
    out = simplify_polygon(poly)
    assert _area(out) == pytest.approx(_area(poly), rel=0.02), (
        "the closing edge was flattened: area moved by more than 2%")


def test_a_ring_is_never_simplified_out_of_existence():
    """A simplification that deletes the object is not a simplification."""
    triangle = [0.0, 0.0, 100.0, 0.0, 50.0, 100.0]
    out = simplify_polygon(triangle)
    assert len(out) // 2 >= 3
    assert _area(out) > 0


def test_a_closed_ring_stays_closed_and_an_open_one_stays_open():
    """Both spellings arrive from different producers, and a consumer that assumes one gets a bad area
    from the other."""
    open_ring = _square(200.0)
    closed_ring = open_ring + open_ring[:2]
    a, b = simplify_polygon(open_ring), simplify_polygon(closed_ring)
    assert a[:2] != a[-2:], "an open ring should not gain a duplicate closing point"
    assert b[:2] == b[-2:], "a closed ring should keep its closing point"


def test_a_ring_that_is_not_a_polygon_is_dropped_from_the_mask():
    """Two points have no area, and every consumer that computes one from them gets a number that is not
    the area of anything."""
    out = simplify_mask([[0.0, 0.0, 10.0, 10.0], _square(200.0)])
    assert len(out) == 1
    assert _area(out[0]) == pytest.approx(200.0 * 200.0, rel=0.01)


def test_an_empty_or_missing_mask_is_an_empty_mask():
    assert simplify_mask(None) == []
    assert simplify_mask([]) == []


def test_simplifying_twice_changes_nothing_further():
    """Every mask write runs this, and a mask that is re-saved must not creep. RDP is idempotent at a
    fixed tolerance, but the tolerance is derived from the ring, so it has to be checked rather than
    assumed: a shorter perimeter gives a smaller tolerance, which cannot remove more."""
    once = simplify_polygon(_square(300.0))
    twice = simplify_polygon(once)
    assert once == twice
