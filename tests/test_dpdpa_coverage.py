"""What the DPDPA gate can actually see: whether a blur covered anything, not whether a row exists.

The gate has always checked that every exported frame carries a PiiAudit row. recheck.py records why that
is not enough - 82.4% of frames holding an annotated person have zero faces redacted, and every one of them
has an audit row, so the gate passed all of them.

These cover the predicate that replaces row-existence, and in particular the two ways it can be wrong in
opposite directions: refusing an empty highway that was correctly found to hold nothing, or passing a frame
with three pedestrians and one blurred face.

Pure: no database, no detector, no image.
"""

from __future__ import annotations

from services.anonymize.coverage import (
    COVERED_FRACTION,
    MARGIN,
    Demand,
    crop_window,
    demands,
    inside_fraction,
    summarize,
    unmet,
)

W, H = 1920, 1080

# A pedestrian standing in the middle of the frame, and the face of the person standing in it.
PERSON = [600.0, 300.0, 760.0, 900.0]
FACE_IN_PERSON = [650.0, 330.0, 720.0, 410.0]


def _row(bbox, l1, name, l0="object"):
    return (bbox, l0, l1, name)


class TestWhatTheAnnotationsDemand:
    def test_a_pedestrian_demands_a_face_and_not_a_plate(self):
        d = demands([_row(PERSON, "vru", "pedestrian")], W, H)
        assert [x.target for x in d] == ["face"]

    def test_a_motorcycle_demands_both(self):
        # two_wheeler is in both group sets: a rider's head sits inside the vehicle box as often as their own.
        d = demands([_row([100, 100, 300, 500], "two_wheeler", "motorcycle")], W, H)
        assert sorted(x.target for x in d) == ["face", "plate"]

    def test_a_truck_demands_a_plate_and_not_a_face(self):
        d = demands([_row([100, 100, 600, 700], "heavy", "truck")], W, H)
        assert [x.target for x in d] == ["plate"]

    def test_a_traffic_sign_demands_nothing(self):
        # l0 is not "object", so it is not a thing that carries PII.
        d = demands([_row([10, 10, 60, 60], "sign", "speed_limit", l0="infra")], W, H)
        assert d == []

    def test_a_cone_demands_nothing(self):
        d = demands([_row([10, 10, 60, 60], "temporary", "cone")], W, H)
        assert d == []

    def test_a_speck_of_an_annotation_demands_nothing(self):
        # Below MIN_CROP_PX the re-check cannot look inside it either, and a demand nothing can satisfy is
        # a refusal with no route out.
        assert demands([_row([10, 10, 18, 18], "vru", "pedestrian")], W, H) == []

    def test_the_window_matches_the_one_the_recheck_looks_in(self):
        # If these drifted apart, the gate could refuse a frame for a region the remediation never examines.
        from services.anonymize import recheck

        assert recheck._crop_window is crop_window
        assert recheck._MARGIN == MARGIN
        assert recheck._FACE_GROUPS == {"vru", "two_wheeler", "three_wheeler"}


class TestWhetherABlurCoveredIt:
    def test_a_face_region_inside_the_person_covers_them(self):
        d = demands([_row(PERSON, "vru", "pedestrian")], W, H)
        assert unmet(d, [{"type": "face", "bbox": FACE_IN_PERSON}]) == []

    def test_containment_not_iou(self):
        """The whole design, asserted directly.

        A real face is a small part of the person carrying it. Using IoU between the annotation and the
        redaction - the metric recheck uses for candidate-vs-stored at the same scale - would score this
        pair at about 0.05 and report every frame in the corpus as uncovered.
        """
        window = crop_window(PERSON, W, H)
        assert window is not None
        fx1, fy1, fx2, fy2 = FACE_IN_PERSON
        wx1, wy1, wx2, wy2 = window
        inter = (min(fx2, wx2) - max(fx1, wx1)) * (min(fy2, wy2) - max(fy1, wy1))
        union = ((fx2 - fx1) * (fy2 - fy1)) + ((wx2 - wx1) * (wy2 - wy1)) - inter
        iou = inter / union
        assert iou < 0.10, "premise of this test is wrong if a face overlaps a person box substantially"
        assert inside_fraction(FACE_IN_PERSON, window) == 1.0
        assert unmet(demands([_row(PERSON, "vru", "pedestrian")], W, H),
                     [{"type": "face", "bbox": FACE_IN_PERSON}]) == []

    def test_a_face_region_on_the_other_side_of_the_frame_covers_nobody(self):
        d = demands([_row(PERSON, "vru", "pedestrian")], W, H)
        assert len(unmet(d, [{"type": "face", "bbox": [20, 20, 90, 100]}])) == 1

    def test_a_plate_region_does_not_satisfy_a_face_demand(self):
        d = demands([_row(PERSON, "vru", "pedestrian")], W, H)
        assert len(unmet(d, [{"type": "plate", "bbox": FACE_IN_PERSON}])) == 1

    def test_one_region_cannot_cover_three_pedestrians(self):
        rows = [_row([600 + i * 200, 300, 760 + i * 200, 900], "vru", "pedestrian") for i in range(3)]
        d = demands(rows, W, H)
        assert len(d) == 3
        missing = unmet(d, [{"type": "face", "bbox": FACE_IN_PERSON}])
        assert len(missing) == 2, "matching must be one-to-one; one blur is not three blurs"

    def test_a_malformed_stored_region_is_not_evidence(self):
        d = demands([_row(PERSON, "vru", "pedestrian")], W, H)
        for bad in ([{"type": "face"}], [{"type": "face", "bbox": []}], [{"type": "face", "bbox": [1, 2]}]):
            assert len(unmet(d, bad)) == 1, bad

    def test_a_partial_overlap_below_the_threshold_does_not_count(self):
        window = crop_window(PERSON, W, H)
        assert window is not None
        # A region mostly outside the person: only a sliver sits within.
        sliver = [window[0] - 100.0, window[1] - 100.0, window[0] + 10.0, window[1] + 10.0]
        assert inside_fraction(sliver, window) < COVERED_FRACTION


class TestTheEmptyHighway:
    """The regression that matters most: a gate keyed on n_faces > 0 refuses every clean frame."""

    def test_a_frame_with_no_annotations_has_no_unmet_demand(self):
        assert unmet(demands([], W, H), []) == []

    def test_a_zero_count_audit_on_an_unannotated_frame_is_clean(self):
        # recheck.py deliberately writes n_faces=0 / regions=[] for "checked, found nothing", so that a
        # checked frame is not mistaken for an unchecked one. That row must not become a refusal.
        assert unmet(demands([], W, H), []) == []

    def test_only_infra_annotations_are_clean(self):
        rows = [_row([10, 10, 60, 60], "temporary", "cone"),
                _row([70, 70, 120, 120], "sign", "speed_limit", l0="infra")]
        assert unmet(demands(rows, W, H), []) == []


class TestTheReportedShape:
    def test_it_counts_by_target_and_names_the_classes(self):
        missing = [Demand("face", (0, 0, 10, 10), "pedestrian"),
                   Demand("face", (0, 0, 10, 10), "rider"),
                   Demand("plate", (0, 0, 10, 10), "truck")]
        s = summarize(missing)
        assert s["missing"] == {"face": 2, "plate": 1}
        assert s["classes"] == ["pedestrian", "rider", "truck"]
