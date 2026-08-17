"""Reanalyse: the label half, and the property that makes the whole action safe to run unattended.

The bounds check is the one rule here with no prior implementation. Five call sites in this codebase clamp a
box to the frame silently and `attribute_agent` computes the out-of-frame fraction only to store it as a
truncation attribute, so a box mostly outside its own image had never been surfaced to anybody.

The rest of the checks are composed from sweeps that already existed and are tested where they live; what is
tested here is the composition: that the action proposes rather than applies, that a frame's findings are
capped with the drop counted rather than hidden, and that a human's label is never questioned.
"""

from __future__ import annotations

import inspect
import uuid

from services.agent.reanalyze import (
    _MACHINE,
    _MAX_FINDINGS_PER_FRAME,
    FINDING_KIND,
    _persist,
    out_of_bounds,
    reanalyze_frame,
)


class TestABoxThatLeavesTheImage:
    def test_a_box_inside_the_frame_is_not_a_finding(self):
        assert out_of_bounds([100, 100, 200, 300], 1920, 1080) is None

    def test_a_box_flush_against_the_edge_is_not_a_finding(self):
        assert out_of_bounds([0, 0, 1920, 1080], 1920, 1080) is None

    def test_a_sub_pixel_overhang_is_a_rounding_artifact_not_an_error(self):
        """Boxes are stored in pixels and a detector legitimately puts an edge a fraction past the boundary.
        Flagging those would bury the real ones."""
        assert out_of_bounds([-0.4, 0, 200, 300], 1920, 1080) is None

    def test_a_truncated_object_at_the_edge_is_normal(self):
        """A pedestrian half out of frame is an ordinary label, already recorded as a truncation attribute.
        Only a box that is mostly outside is a coordinate error."""
        assert out_of_bounds([-20, 100, 180, 300], 1920, 1080) is None

    def test_a_box_mostly_outside_the_image_is_flagged(self):
        found = out_of_bounds([-500, 10, 100, 200], 1920, 1080)
        assert found is not None
        score, reason = found
        assert 0.0 < score <= 1.0
        assert "outside" in reason and "1920x1080" in reason

    def test_it_flags_every_edge(self):
        for bbox in ([-500, 10, 100, 200], [1900, 10, 2500, 200],
                     [10, -500, 200, 100], [10, 1000, 200, 1600]):
            assert out_of_bounds(bbox, 1920, 1080) is not None, bbox

    def test_a_corner_overhang_is_not_double_counted(self):
        """Summing the overhangs would count the corner twice and report more than 100% outside. The check
        clips and compares areas instead."""
        found = out_of_bounds([-10, -10, 400, 400], 1920, 1080)
        assert found is None, "a box 95% inside the image was called out of bounds"

    def test_a_degenerate_box_is_left_to_the_rule_that_owns_it(self):
        """A zero-area box is a finding of its own (min_box_size); reporting it here as well would put two
        rows on the queue for one mistake."""
        assert out_of_bounds([100, 100, 100, 100], 1920, 1080) is None

    def test_an_unknown_frame_size_is_not_guessed(self):
        assert out_of_bounds([-500, 10, 100, 200], 0, 0) is None


class TestItProposesAndNeverApplies:
    def test_the_action_never_writes_an_object(self):
        """The safety argument for running this over the whole corpus unattended. Auto-applying a class
        change is what put 1,047 buses inside a bus shelter, so this half only ever queues.
        """
        src = inspect.getsource(__import__("services.agent.reanalyze", fromlist=["x"]))
        for forbidden in ("db.delete(", ".class_id =", ".state =", ".bbox =", ".attrs ="):
            assert forbidden not in src, (
                f"reanalyze mutates a label ({forbidden}); its findings must go to the review queue "
                f"instead, and the corpus sweep is only safe to run unattended because they do")

    def test_findings_carry_one_kind_that_fits_the_column(self):
        """ErrorCandidate.kind is String(24), and a kind per rule would split the precision measurement into
        samples too small to measure. The rule goes in the detail, as policy_violation already does."""
        assert len(FINDING_KIND) <= 24

    def test_human_labels_are_out_of_scope(self):
        """A person looked at it, which is more evidence than any check here can offer."""
        assert "human" not in _MACHINE
        assert set(_MACHINE) == {"fused", "auto_accept", "interpolated"}


class TestTheCapIsStatedNotHidden:
    def test_a_frame_cannot_flood_the_queue(self):
        """Measured on six live frames: 8, 10, 0, 27, 26 and 30 findings, mostly one track-level fact
        restated per object. Thirty rows across 33,547 frames is a million candidates."""
        assert 0 < _MAX_FINDINGS_PER_FRAME <= 20

    def test_the_dropped_count_is_reported(self):
        """A queue that silently discards two thirds of what it found reads as a clean frame."""
        sig = inspect.getsource(reanalyze_frame)
        assert "findings_dropped" in sig


class TestPersistenceIsScopedToTheFrame:
    def test_it_clears_only_this_frames_candidates(self):
        """`run_detection` clears every pending candidate of the kinds it runs, which is right for a corpus
        sweep and wrong here: pressing the button on one frame must not discard a hundred other frames'
        pending findings."""
        src = inspect.getsource(_persist)
        assert "Object.frame_id == frame_id" in src
        assert "status" in src, "a human's verdict would be deleted along with the pending rows"


class TestTheRedactionHalfIsTheOneThatApplies:
    def test_the_plan_form_writes_nothing(self):
        """A blur cannot be undone: the unredacted original is deliberately never stored, so the operation
        has to be inspectable before it is taken."""
        src = inspect.getsource(reanalyze_frame)
        assert "apply=apply" in src, "the plan form would blur the frame it was asked to describe"
        assert "if apply" in src


def test_a_finding_is_shaped_like_every_other_queue_candidate():
    """The queue ranks across detectors, so a finding that skipped the shared scale would rank against
    calibrated detectors on a scale of its own."""
    from services.agent.reanalyze import _finding

    f = _finding(uuid.uuid4(), "out_of_bounds", 5.0, "reason")
    assert set(f) == {"object_id", "kind", "score", "proposed_label", "detail"}
    assert f["score"] == 1.0, "a badly scaled rule was not clamped onto the shared suspicion scale"
    assert f["detail"] == {"rule": "out_of_bounds", "reason": "reason"}
