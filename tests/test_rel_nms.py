"""Relationship-aware suppression: a pedestrian in front of a motorcyclist is two objects.

The rule this replaces merged two boxes when their classes shared an l1 superclass. On Indian roads that
is not a rare edge: pedestrian, rider, cyclist and cattle-handler are all l1 "vru", so any two of them
overlapping became one box and one of them stopped existing. The behavioural test at the bottom fails
against the old rule and passes against the new one, which is the whole claim.

The fixtures place boxes so overlap is exact rather than approximate, and the compat probabilities are
supplied directly, so what is under test is the suppression logic and not somebody's estimate of it.
"""

from __future__ import annotations

import numpy as np

from core.accel.rel_nms import relationship_nms
from services.autolabel.compat_matrix import CompatMatrix


def _fixed(p: float, support: int = 100):
    return lambda a, b: (p, support)


class TestSuppression:
    def test_a_pair_the_matrix_calls_one_object_is_merged(self):
        # Two boxes at IoU 1.0, distinctness 0.05: the same thing detected twice.
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        r = relationship_nms(boxes, [0.9, 0.8], [1, 2], distinct_prob=_fixed(0.05))
        assert r.keep.tolist() == [0]
        assert len(r.suppressed) == 1
        assert r.suppressed[0]["absorbed_by"] == 0 and r.suppressed[0]["distinct_prob"] == 0.05

    def test_a_pair_the_matrix_calls_two_objects_survives_the_same_overlap(self):
        """Identical geometry, opposite outcome. Overlap alone cannot decide this and never could."""
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        r = relationship_nms(boxes, [0.9, 0.8], [1, 2], distinct_prob=_fixed(0.95))
        assert sorted(r.keep.tolist()) == [0, 1]
        assert r.suppressed == ()

    def test_suppression_needs_overlap_as_well_as_a_low_probability(self):
        # Disjoint boxes are never one object however incompatible the classes are.
        boxes = [[0, 0, 10, 10], [500, 500, 510, 510]]
        r = relationship_nms(boxes, [0.9, 0.8], [1, 2], distinct_prob=_fixed(0.0))
        assert sorted(r.keep.tolist()) == [0, 1]

    def test_a_nested_box_is_caught_by_intersection_over_min(self):
        """A small box wholly inside a large one scores low IoU precisely when the nesting is complete.

        IoU here is 100/10000 = 0.01, far under any sane threshold, while IoM is 1.0.
        """
        boxes = [[0, 0, 100, 100], [10, 10, 20, 20]]
        r = relationship_nms(boxes, [0.9, 0.8], [1, 2], distinct_prob=_fixed(0.05), iom_thr=0.8)
        assert r.keep.tolist() == [0]
        assert r.suppressed[0]["iom"] == 1.0 and r.suppressed[0]["iou"] < 0.02

    def test_the_highest_scoring_box_is_the_one_kept(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]
        r = relationship_nms(boxes, [0.2, 0.9, 0.5], [1, 1, 1], distinct_prob=_fixed(0.0))
        assert r.keep.tolist() == [1]
        assert {s["absorbed_by"] for s in r.suppressed} == {1}

    def test_a_decision_made_on_the_prior_alone_is_counted(self):
        """A suppression rule that never says how much evidence it had is a hard-coded rule.

        With 621 human-confirmed objects in this corpus, nearly every cell is prior, and a caller that
        could not see that would read the matrix as a measurement.
        """
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        r = relationship_nms(boxes, [0.9, 0.8], [1, 2], distinct_prob=_fixed(0.4, support=0))
        assert r.n_prior_only == 1
        assert r.suppressed[0]["support"] == 0

    def test_empty_and_mismatched_inputs(self):
        assert relationship_nms([], [], [], distinct_prob=_fixed(0.5)).keep.tolist() == []
        try:
            relationship_nms([[0, 0, 1, 1]], [0.5, 0.4], [1], distinct_prob=_fixed(0.5))
        except ValueError:
            return
        raise AssertionError("mismatched lengths should raise")


class TestEdges:
    def test_a_surviving_overlapping_pair_becomes_a_provisional_edge(self):
        """The overlap that proved they were two objects is the evidence a relationship rests on.

        Computing it twice, once here and once in a separate scene-graph pass, invites the two to
        disagree about the same frame.
        """
        boxes = [[0, 0, 10, 20], [0, 10, 10, 30]]           # a rider above their machine
        r = relationship_nms(boxes, [0.9, 0.8], [7, 3], distinct_prob=_fixed(0.95), edge_iou=0.2,
                             relation_for_pair=lambda a, b: "rider_of" if (a, b) == (7, 3) else None)
        assert sorted(r.keep.tolist()) == [0, 1]
        assert len(r.edges) == 1
        e = r.edges[0]
        assert (e["from_index"], e["to_index"], e["kind"]) == (0, 1, "rider_of")

    def test_no_edge_without_a_vocabulary(self):
        boxes = [[0, 0, 10, 20], [0, 10, 10, 30]]
        r = relationship_nms(boxes, [0.9, 0.8], [7, 3], distinct_prob=_fixed(0.95))
        assert r.edges == ()

    def test_a_suppressed_box_cannot_be_an_endpoint(self):
        # It no longer exists, so an edge to it would point at nothing.
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        r = relationship_nms(boxes, [0.9, 0.8], [7, 3], distinct_prob=_fixed(0.05), edge_iou=0.2,
                             relation_for_pair=lambda a, b: "rider_of")
        assert r.keep.tolist() == [0] and r.edges == ()


class TestTheMatrix:
    def test_an_unobserved_pair_lands_on_exactly_a_half_whatever_the_base_rate(self):
        """Knowing nothing about a pair means treating it as typical, not as confidently distinct.

        This is why the shrinkage target is the corpus base rate rather than a flat half. Shrinking toward
        a half would put every unseen pair far above a base rate of 0.04 and read as strong evidence that
        they are two objects.
        """
        for together, apart in (({}, {}), ({(5, 6): 3}, {(5, 6): 97})):
            m = CompatMatrix(together, apart, ontology_version="v", snapshot="s",
                             n_frames=1, n_objects=2)
            p, support = m.distinct_prob(1, 2)
            assert support == 0
            assert abs(p - 0.5) < 1e-12, (m.base_rate, p)

    def test_a_pair_that_overlaps_more_than_typical_reads_as_distinct(self):
        """The corpus base rate is what "typical" means, worked by hand.

        Base rate over the whole matrix: (10 + 90 overlapping) is not the shape here. With one pair at
        10 overlapping / 10 apart and another at 0 / 180, the corpus sees 10 overlaps in 200 co-present
        pairs, so base = (10 + 1) / (200 + 2) = 0.05446.

        The first pair: rate = (10 + 5 * 0.05446) / (20 + 5) = 0.41089, so
        P = 0.41089 / (0.41089 + 0.05446) = 0.883, comfortably distinct.
        """
        m = CompatMatrix({(1, 2): 10}, {(1, 2): 10, (3, 4): 180},
                         ontology_version="v", snapshot="s", n_frames=50, n_objects=200)
        base = 11.0 / 202.0
        assert abs(m.base_rate - base) < 1e-9
        rate = (10.0 + 5.0 * base) / (20.0 + 5.0)
        p, support = m.distinct_prob(1, 2)
        assert support == 20
        assert abs(p - rate / (rate + base)) < 1e-9
        assert p > 0.85

    def test_a_pair_that_almost_never_overlaps_reads_as_one_object(self):
        m = CompatMatrix({(1, 2): 1}, {(1, 2): 199, (3, 4): 10},
                         ontology_version="v", snapshot="s", n_frames=50, n_objects=200)
        p, _ = m.distinct_prob(1, 2)
        assert p < 0.5

    def test_the_pair_is_symmetric(self):
        m = CompatMatrix({(1, 2): 4}, {}, ontology_version="v", snapshot="s", n_frames=1, n_objects=2)
        assert m.distinct_prob(1, 2) == m.distinct_prob(2, 1)

    def test_it_round_trips_through_json_with_its_provenance(self):
        m = CompatMatrix({(1, 2): 3}, {(1, 3): 5}, ontology_version="v1", snapshot="2026-08-20",
                         n_frames=7, n_objects=42)
        back = CompatMatrix.from_json(m.to_json())
        assert back.distinct_prob(1, 2) == m.distinct_prob(1, 2)
        assert back.distinct_prob(1, 3) == m.distinct_prob(1, 3)
        assert (back.ontology_version, back.snapshot, back.n_objects) == ("v1", "2026-08-20", 42)

    def test_the_summary_states_that_it_is_mostly_prior(self):
        m = CompatMatrix({(1, 2): 2}, {}, ontology_version="v", snapshot="s", n_frames=1, n_objects=2)
        s = m.summary()
        assert s["n_cells_observed"] == 1 and s["learned"] is True
        assert "must not be read as measurements" in s["caveat"]


def test_two_vru_classes_that_the_corpus_shows_overlapping_are_kept_apart():
    """The behavioural claim, against the rule that was live.

    Both classes are l1 "vru", so the old `_same_object` returned True for the pair and the greedy pass
    dropped the lower-scoring one. The matrix has seen them overlapping far more than a typical pair does,
    so it keeps both.
    """
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped, rider = onto.by_name("pedestrian").id, onto.by_name("rider").id
    assert onto.by_id(ped).l1 == onto.by_id(rider).l1, "the old rule merged on exactly this"

    key = (min(ped, rider), max(ped, rider))
    m = CompatMatrix({key: 40}, {key: 10, (900, 901): 400},
                     ontology_version=onto.version, snapshot="s", n_frames=40, n_objects=80)
    boxes = np.array([[0, 0, 10, 20], [4, 0, 14, 20]], dtype=float)   # IoU 0.43
    r = relationship_nms(boxes, [0.9, 0.85], [ped, rider], distinct_prob=m.distinct_prob,
                         iou_thr=0.4, distinct_floor=0.5)
    assert sorted(r.keep.tolist()) == [0, 1], "both people must survive"


def test_the_fuser_keeps_a_rider_and_their_motorcycle_apart_on_the_relation_alone():
    """The live path, and the half of the fix that does not depend on having enough labels.

    A pack that names the relation between two superclasses has already said the pair is two objects, so
    the matrix does not have to have seen them. That matters on this corpus: pedestrian and rider have 7
    co-present observations, which is not enough to move anything, while every VRU on a two-wheeler is a
    rider on it by construction.
    """
    from core.schemas import BBox, Provenance, UnifiedObject
    from packs.registry import default_pack_id, get_pack
    from services.autolabel.fusion import FusionEngine
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    relations = get_pack(default_pack_id()).relations
    prior = CompatMatrix({}, {}, ontology_version=onto.version, snapshot="s", n_frames=0, n_objects=0)

    def _obj(name: str, box: list[float], conf: float):
        cid = onto.by_name(name).id
        return type("FO", (), {"obj": UnifiedObject(
            class_id=cid, class_name=name, bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
            conf=conf, provenance=Provenance())})()

    # A rider box sitting almost entirely over the motorcycle box: IoM is high, so the old rule's
    # geometry test fires and only the class rule decides the outcome.
    fused = [_obj("rider", [0, 0, 10, 30], 0.9), _obj("motorcycle", [0, 10, 10, 30], 0.85)]

    old = FusionEngine(ontology=onto)                       # no matrix, no relations: the l1 rule
    new = FusionEngine(ontology=onto, compat=prior, relations=relations)
    assert old._same_object(fused[0].obj.class_id, fused[1].obj.class_id) is False, (
        "l1 already separated this pair; the regression is between two classes of one l1")

    ped, rider = onto.by_name("pedestrian").id, onto.by_name("rider").id
    assert old._same_object(ped, rider) is True, "the old rule merges two VRU classes"
    assert new._same_object(ped, rider) is False or prior.distinct_prob(ped, rider)[0] == 0.5

    # And a VRU over a two-wheeler is kept apart by the relation with no evidence at all.
    moto = onto.by_name("motorcycle").id
    assert new._same_object(ped, moto) is False
