"""Lane topology: the layer that made the HD map exports into maps rather than piles of lines.

A Lanelet2 file is defined by `relation type=lanelet` naming a left and a right boundary. The exporter
emitted the boundaries as loose ways and no relations, so nothing downstream could route on it. OpenDRIVE
roads were emitted with one centre lane of `type="none"` and no driving lanes at all. Both formats needed
the same two facts a pile of polylines does not carry: which boundaries bound the same lane, and which
lane follows which.

These are the properties that decide whether the pairing can be trusted. Each one is a way the naive
version of this gets it wrong.
"""

from __future__ import annotations

import math

from services.hdmap.topology import (
    MAX_LANE_W_M,
    arclength,
    build_lanelets,
    chain,
    overlap_fraction,
    resample,
    signed_offset,
    to_local,
    to_world,
)

LON0, LAT0 = 77.5946, 12.9716
M_PER_DEG_LAT = 111320.0


def _line(east0: float, north0: float, east1: float, north1: float, n: int = 9) -> list[tuple[float, float]]:
    """A straight boundary in local metres, returned as (lon, lat)."""
    pts = [(east0 + (east1 - east0) * i / (n - 1), north0 + (north1 - north0) * i / (n - 1)) for i in range(n)]
    return to_world(pts, (LON0, LAT0))


class TestGeometryPrimitives:
    def test_local_and_world_round_trip(self):
        line = _line(0.0, 0.0, 3.0, 40.0)
        back = to_world(to_local(line, (LON0, LAT0)), (LON0, LAT0))
        for (a, b), (c, d) in zip(line, back, strict=True):
            assert abs(a - c) < 1e-9 and abs(b - d) < 1e-9

    def test_resample_keeps_the_ends_and_the_length(self):
        line = to_local(_line(0.0, 0.0, 0.0, 50.0), (LON0, LAT0))
        r = resample(line, 17)
        assert len(r) == 17
        assert r[0] == line[0] and r[-1] == line[-1]
        assert abs(arclength(r) - arclength(line)) < 1e-6

    def test_resample_makes_two_differently_traced_boundaries_comparable(self):
        """Boundaries share no vertices, so a vertex-to-vertex comparison compares their spacing.

        Here the two sides of one 3 m lane are traced with 3 and 40 points. Unresampled, index i on one
        is nowhere near index i on the other; resampled, every pair is 3 m apart.
        """
        left = to_local(_line(0.0, 0.0, 0.0, 60.0, n=3), (LON0, LAT0))
        right = to_local(_line(3.0, 0.0, 3.0, 60.0, n=40), (LON0, LAT0))
        off, agree = signed_offset(left, right)
        assert abs(abs(off) - 3.0) < 0.05
        assert agree == 1.0

    def test_offset_sign_says_which_side(self):
        a = to_local(_line(0.0, 0.0, 0.0, 40.0), (LON0, LAT0))
        right_of_a = to_local(_line(3.2, 0.0, 3.2, 40.0), (LON0, LAT0))
        left_of_a = to_local(_line(-3.2, 0.0, -3.2, 40.0), (LON0, LAT0))
        assert signed_offset(a, right_of_a)[0] < 0
        assert signed_offset(a, left_of_a)[0] > 0

    def test_crossing_boundaries_do_not_agree_on_a_side(self):
        """A crossing pair has a small mean offset and would pass a test that only looked at the mean."""
        a = to_local(_line(0.0, 0.0, 0.0, 40.0), (LON0, LAT0))
        x = to_local(_line(-3.0, 0.0, 3.0, 40.0), (LON0, LAT0))
        off, agree = signed_offset(a, x)
        assert abs(off) < MAX_LANE_W_M, "the mean offset alone looks like a plausible lane width"
        assert agree < 0.9, "but the sign changes along the length, which is what rejects it"

    def test_consecutive_stretches_barely_overlap(self):
        a = to_local(_line(0.0, 0.0, 0.0, 30.0), (LON0, LAT0))
        after = to_local(_line(0.0, 31.0, 0.0, 60.0), (LON0, LAT0))
        beside = to_local(_line(3.0, 0.0, 3.0, 30.0), (LON0, LAT0))
        assert overlap_fraction(a, after) < 0.1
        assert overlap_fraction(a, beside) > 0.9


class TestPairing:
    def test_two_boundaries_one_lane_width_apart_become_one_lanelet(self):
        out = build_lanelets([
            {"id": "L", "points": _line(0.0, 0.0, 0.0, 50.0), "lane_type": "solid", "confidence": 0.9},
            {"id": "R", "points": _line(3.4, 0.0, 3.4, 50.0), "lane_type": "dashed", "confidence": 0.7},
        ])
        assert len(out["lanelets"]) == 1
        ll = out["lanelets"][0]
        assert abs(ll.width_m - 3.4) < 0.1
        assert out["unpaired"] == []
        # A lane is no better placed than the worse of the two traces that bound it.
        assert ll.confidence == 0.7
        assert set(ll.source_elements) == {"L", "R"}
        # The centreline sits midway between the two boundaries. Checked against the boundaries rather
        # than an absolute coordinate, because the local frame's origin is the centroid of the input.
        o = out["origin"]
        left, right, mid = (to_local(list(x), o) for x in (ll.left, ll.right, ll.centre))
        for (lx, ly), (rx, ry), (mx, my) in zip(left, right, mid, strict=True):
            assert abs(mx - (lx + rx) / 2.0) < 1e-6 and abs(my - (ly + ry) / 2.0) < 1e-6

    def test_the_left_boundary_is_actually_on_the_left(self):
        out = build_lanelets([
            {"id": "west", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.8},
            {"id": "east", "points": _line(3.2, 0.0, 3.2, 50.0), "confidence": 0.8},
        ])
        ll = out["lanelets"][0]
        left = to_local(list(ll.left), out["origin"])
        right = to_local(list(ll.right), out["origin"])
        # Travelling north, the western boundary is on the left.
        assert left[0][0] < right[0][0]

    def test_a_shared_boundary_bounds_two_lanes(self):
        """Adjacent lanes share a marking, so the middle boundary must serve both."""
        out = build_lanelets([
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 60.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.3, 0.0, 3.3, 60.0), "confidence": 0.9},
            {"id": "c", "points": _line(6.6, 0.0, 6.6, 60.0), "confidence": 0.9},
        ])
        assert len(out["lanelets"]) == 2, "three boundaries make two lanes, not one"
        assert out["unpaired"] == []
        shared = [ll for ll in out["lanelets"] if "b" in ll.source_elements]
        assert len(shared) == 2

    def test_a_boundary_too_far_to_bound_a_lane_is_reported_not_paired(self):
        out = build_lanelets([
            {"id": "near1", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
            {"id": "near2", "points": _line(3.1, 0.0, 3.1, 50.0), "confidence": 0.9},
            {"id": "far", "points": _line(40.0, 0.0, 40.0, 50.0), "confidence": 0.9},
        ])
        assert len(out["lanelets"]) == 1
        assert [u["id"] for u in out["unpaired"]] == ["far"]
        assert "outside one lane's width" in out["unpaired"][0]["reason"]

    def test_a_cross_street_is_not_paired_with_the_road_it_crosses(self):
        out = build_lanelets([
            {"id": "along", "points": _line(0.0, 0.0, 0.0, 60.0), "confidence": 0.9},
            {"id": "across", "points": _line(-30.0, 30.0, 30.0, 30.0), "confidence": 0.9},
        ])
        assert out["lanelets"] == []
        assert len(out["unpaired"]) == 2

    def test_a_single_boundary_cannot_make_a_lane(self):
        out = build_lanelets([{"id": "only", "points": _line(0.0, 0.0, 0.0, 40.0)}])
        assert out["lanelets"] == []
        assert out["unpaired"] == [{"id": "only", "reason": "no other boundary to pair with"}]

    def test_a_degenerate_boundary_is_dropped_with_its_reason(self):
        out = build_lanelets([
            {"id": "point", "points": [(LON0, LAT0)]},
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 40.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.2, 0.0, 3.2, 40.0), "confidence": 0.9},
        ])
        assert len(out["lanelets"]) == 1
        assert {"id": "point", "reason": "fewer than two points"} in out["unpaired"]


class TestChaining:
    def test_a_lane_feeds_the_lane_ahead_of_it(self):
        first = build_lanelets([
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 40.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.2, 0.0, 3.2, 40.0), "confidence": 0.9},
        ])["lanelets"]
        second = build_lanelets([
            {"id": "c", "points": _line(0.0, 41.0, 0.0, 80.0), "confidence": 0.9},
            {"id": "d", "points": _line(3.2, 41.0, 3.2, 80.0), "confidence": 0.9},
        ])["lanelets"]
        assert chain(first + second) == [(0, 1)]

    def test_the_oncoming_lane_is_not_a_successor(self):
        """At a junction the ends of opposing lanes meet. Distance alone would join them."""
        north = build_lanelets([
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 40.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.2, 0.0, 3.2, 40.0), "confidence": 0.9},
        ])["lanelets"]
        south = build_lanelets([
            {"id": "c", "points": _line(0.0, 80.0, 0.0, 41.0), "confidence": 0.9},
            {"id": "d", "points": _line(3.2, 80.0, 3.2, 41.0), "confidence": 0.9},
        ])["lanelets"]
        # The northbound lane ends where the southbound lane ends, about a metre apart.
        assert chain(north + south) == []

    def test_a_gap_too_wide_to_bridge_is_not_chained(self):
        a = build_lanelets([
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 40.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.2, 0.0, 3.2, 40.0), "confidence": 0.9},
        ])["lanelets"]
        far = build_lanelets([
            {"id": "c", "points": _line(0.0, 90.0, 0.0, 130.0), "confidence": 0.9},
            {"id": "d", "points": _line(3.2, 90.0, 3.2, 130.0), "confidence": 0.9},
        ])["lanelets"]
        assert chain(a + far) == []


def test_a_curve_is_paired_as_one_lane():
    """Two boundaries traced round the same bend at different radii must still pair."""
    # Concentric about a common centre at (0, 100), which is what the two edges of one lane on a bend
    # actually are. Two arcs sharing a start point instead diverge, and are not a lane.
    def arc(radius: float, n: int = 15) -> list[tuple[float, float]]:
        pts = [(radius * math.sin(t), 100.0 - radius * math.cos(t))
               for t in (i * (math.pi / 6) / (n - 1) for i in range(n))]
        return to_world(pts, (LON0, LAT0))

    out = build_lanelets([
        {"id": "inner", "points": arc(100.0), "confidence": 0.8},
        {"id": "outer", "points": arc(103.3), "confidence": 0.8},
    ])
    assert len(out["lanelets"]) == 1, out["unpaired"]
    assert 2.0 <= out["lanelets"][0].width_m <= 5.5


def _fused_line(name, east0, north0, east1, north1, lane_type="solid", conf=0.9, n=9):
    pts = _line(east0, north0, east1, north1, n=n)
    return {"element_id": name, "kind": "lane", "confidence": conf,
            "attrs": {"lane_type": lane_type},
            "wkt": "LINESTRING(" + ", ".join(f"{lo} {la}" for lo, la in pts) + ")"}


class TestExportsAreRoutable:
    """The old exporter's only test asserted the XML parsed. Both files parsed and neither was a map."""

    def _two_lane_road(self):
        return [
            _fused_line("b1", 0.0, 0.0, 0.0, 50.0, "solid"),
            _fused_line("b2", 3.3, 0.0, 3.3, 50.0, "dashed"),
            _fused_line("b3", 6.6, 0.0, 6.6, 50.0, "solid"),
        ]

    def test_lanelet2_emits_lanelet_relations(self):
        """A Lanelet2 file is defined by these relations. Without them nothing can route on it."""
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_lanelet2_osm

        root = ET.fromstring(to_lanelet2_osm(self._two_lane_road()))
        rels = [r for r in root.findall("relation")
                if any(t.get("k") == "type" and t.get("v") == "lanelet" for t in r.findall("tag"))]
        assert len(rels) == 2, "three boundaries make two lanes"
        for r in rels:
            roles = sorted(m.get("role") for m in r.findall("member"))
            assert roles == ["left", "right"]
            assert all(m.get("type") == "way" for m in r.findall("member"))
            w = next(t.get("v") for t in r.findall("tag") if t.get("k") == "width")
            assert 2.0 <= float(w) <= 5.5

    def test_the_shared_boundary_is_one_way_referenced_twice(self):
        """Two coincident ways would make the middle marking two markings, and routing would not see
        the lanes as adjacent."""
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_lanelet2_osm

        root = ET.fromstring(to_lanelet2_osm(self._two_lane_road()))
        refs = [m.get("ref") for r in root.findall("relation") for m in r.findall("member")]
        assert len(refs) == 4 and len(set(refs)) == 3, "the middle boundary is shared"

    def test_lanelet2_records_which_lane_feeds_which(self):
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_lanelet2_osm

        fused = [
            _fused_line("a1", 0.0, 0.0, 0.0, 40.0),
            _fused_line("a2", 3.3, 0.0, 3.3, 40.0),
            _fused_line("b1", 0.0, 41.0, 0.0, 80.0),
            _fused_line("b2", 3.3, 41.0, 3.3, 80.0),
        ]
        root = ET.fromstring(to_lanelet2_osm(fused))
        succ = [r for r in root.findall("relation")
                if any(t.get("k") == "type" and t.get("v") == "lane_succession" for t in r.findall("tag"))]
        assert len(succ) == 1
        roles = sorted(m.get("role") for m in succ[0].findall("member"))
        assert roles == ["from", "to"]

    def test_an_unpaired_boundary_is_kept_and_says_why(self):
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_lanelet2_osm

        fused = self._two_lane_road() + [_fused_line("lonely", 60.0, 0.0, 60.0, 50.0)]
        root = ET.fromstring(to_lanelet2_osm(fused))
        orphans = [w for w in root.findall("way")
                   if any(t.get("k") == "lanelet" and t.get("v") == "none" for t in w.findall("tag"))]
        assert len(orphans) == 1
        why = next(t.get("v") for t in orphans[0].findall("tag") if t.get("k") == "unpaired_reason")
        assert "outside one lane's width" in why

    def test_opendrive_roads_have_a_lane_to_drive_in(self):
        """Every road used to carry one centre lane of type "none" and nothing else."""
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_opendrive

        root = ET.fromstring(to_opendrive(self._two_lane_road(), (LAT0, LON0)))
        roads = root.findall("road")
        assert len(roads) == 2
        for road in roads:
            sec = road.find("lanes/laneSection")
            driving = [ln for side in ("left", "right") for ln in sec.findall(f"{side}/lane")
                       if ln.get("type") == "driving"]
            assert len(driving) == 2, "a lane each side of the reference line"
            for ln in driving:
                a = float(ln.find("width").get("a"))
                assert a > 0.5, "a driving lane needs a width"

    def test_opendrive_links_a_road_to_the_road_ahead(self):
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_opendrive

        fused = [
            _fused_line("a1", 0.0, 0.0, 0.0, 40.0),
            _fused_line("a2", 3.3, 0.0, 3.3, 40.0),
            _fused_line("b1", 0.0, 41.0, 0.0, 80.0),
            _fused_line("b2", 3.3, 41.0, 3.3, 80.0),
        ]
        root = ET.fromstring(to_opendrive(fused, (LAT0, LON0)))
        succ = [s for r in root.findall("road") for s in r.findall("link/successor")]
        pred = [p for r in root.findall("road") for p in r.findall("link/predecessor")]
        assert len(succ) == 1 and len(pred) == 1
        assert succ[0].get("elementType") == "road" and pred[0].get("contactPoint") == "end"

    def test_the_marking_type_survives_into_the_road_mark(self):
        import xml.etree.ElementTree as ET

        from services.hdmap.export import to_opendrive

        fused = [_fused_line("l", 0.0, 0.0, 0.0, 50.0, "solid"),
                 _fused_line("r", 3.3, 0.0, 3.3, 50.0, "dashed")]
        root = ET.fromstring(to_opendrive(fused, (LAT0, LON0)))
        marks = {m.get("type") for m in root.findall("road/lanes/laneSection/*/lane/roadMark")}
        assert marks == {"solid", "broken"}, "a dashed marking is 'broken' in OpenDRIVE"


class TestPairingIsScopedToWhatWasSeenTogether:
    """Ungrouped pairing built lanes out of two views of the same marking.

    The georeferencer writes one element per source frame, so consecutive frames each hold their own
    view of a marking, a metre or two apart after ego motion and projection error. On the first real run,
    199 boundaries from a KITTI drive produced 128 lanelets of which 127 paired one frame's marking with
    the next frame's, and the median "lane" came out 2.40 m wide on a road whose lanes are 3.0 to 3.5 m.

    `group` confines pairing to boundaries that appeared in the same image, which is the only evidence
    that two boundaries bound one lane.
    """

    def test_boundaries_from_different_frames_are_not_a_lane(self):
        out = build_lanelets([
            {"id": "f1-left", "group": "frame-1", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
            {"id": "f2-left", "group": "frame-2", "points": _line(2.4, 0.0, 2.4, 50.0), "confidence": 0.9},
        ])
        assert out["lanelets"] == []
        assert all(u["reason"] == "no other boundary was observed alongside this one in the same frame"
                   for u in out["unpaired"])

    def test_the_same_two_boundaries_do_pair_when_seen_together(self):
        """Same geometry, one frame. The grouping is the only thing that changed."""
        out = build_lanelets([
            {"id": "left", "group": "frame-1", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
            {"id": "right", "group": "frame-1", "points": _line(3.3, 0.0, 3.3, 50.0), "confidence": 0.9},
        ])
        assert len(out["lanelets"]) == 1

    def test_ungrouped_input_still_pairs_freely(self):
        """Grouping is opt-in: a caller holding continuous traces rather than per-frame views omits it."""
        out = build_lanelets([
            {"id": "a", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
            {"id": "b", "points": _line(3.3, 0.0, 3.3, 50.0), "confidence": 0.9},
        ])
        assert len(out["lanelets"]) == 1

    def test_an_unpaired_boundary_is_blamed_on_the_most_specific_reason(self):
        """The first version reported whichever rejection happened last.

        With many boundaries in other frames, that meant a boundary was blamed on being alone in its
        frame even where a same-frame candidate existed and had failed on width. The reason a person acts
        on has to be the one that actually applied.
        """
        out = build_lanelets([
            {"id": "a", "group": "f1", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
            {"id": "b", "group": "f1", "points": _line(30.0, 0.0, 30.0, 50.0), "confidence": 0.9},
            {"id": "elsewhere", "group": "f2", "points": _line(0.0, 0.0, 0.0, 50.0), "confidence": 0.9},
        ])
        assert out["lanelets"] == []
        why = {u["id"]: u["reason"] for u in out["unpaired"]}
        assert "outside one lane's width" in why["a"], why
        assert "outside one lane's width" in why["b"], why
        assert why["elsewhere"] == "no other boundary was observed alongside this one in the same frame"
