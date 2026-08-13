"""An annotator could write `accepted` and skip the review step entirely.

The two-step workflow is that an annotator's work goes to a reviewer before it counts, and the client
encodes it (`web/lib/user.ts` `acceptState`). The server did not check: `POST /objects/{id}/review` wrote
`payload.state or _ACTION_STATE[action]`, so the state came from the request body and the caller's role was
never consulted. The QA step was advisory, which means it was not a step.

The frame editor demonstrated the hole by accident. Its save path used `acceptState(role)` while the A key
hardcoded `accepted`, so the same person approving the same box two ways landed it in two different queues.
Fixing the client alone would have left the server open, because the client is not what decides.
"""

from __future__ import annotations

import pytest

from services.review_policy import OBJECT_STATES, ReviewStateError, state_for, was_clamped


class TestAnnotatorsSubmit:
    def test_an_annotators_accept_is_a_submission(self):
        assert state_for("accept", None, "annotator", "review") == "submitted"

    def test_an_annotator_asking_for_accepted_outright_still_submits(self):
        # The editor did exactly this. It is a client that has not been told about the workflow, not an
        # attack, so the request is answered with the correct state rather than refused.
        assert state_for("accept", "accepted", "annotator", "review") == "submitted"

    def test_an_annotator_cannot_write_the_gate_state_either(self):
        # `auto_accept` is what the machine gate writes. A human hand-writing it would launder an unreviewed
        # label into the band that reports how the gate is performing.
        assert state_for(None, "auto_accept", "annotator", "review") == "submitted"

    def test_an_annotator_may_still_reject(self):
        # Rejecting is not an approval and needs no ceiling: it removes a label from consideration.
        assert state_for("reject", None, "annotator", "review") == "rejected"

    def test_an_annotator_may_send_something_back_to_unreviewed(self):
        # The undo path writes this, and it claims nothing.
        assert state_for(None, "review", "annotator", "accepted") == "review"


class TestReviewersAndAbove:
    def test_a_reviewers_accept_counts(self):
        assert state_for("accept", None, "reviewer", "review") == "accepted"

    def test_an_admin_is_at_least_a_reviewer(self):
        assert state_for("accept", None, "admin", "review") == "accepted"

    def test_confirm_is_the_same_decision_as_accept(self):
        assert state_for("confirm", None, "reviewer", "review") == "accepted"


class TestNoRoleAtAll:
    def test_an_unauthenticated_deployment_is_not_treated_as_an_annotator(self):
        """A deployment with auth off has no roles. Clamping there would change behaviour for every
        single-user install based on the absence of a concept rather than on a decision."""
        assert state_for("accept", None, None, "review") == "accepted"


class TestTheEdges:
    def test_an_unknown_state_is_refused_rather_than_guessed(self):
        # A state that is not a state has no correct answer to clamp to.
        with pytest.raises(ReviewStateError):
            state_for(None, "approved", "reviewer", "review")

    def test_every_state_it_can_return_is_a_real_state(self):
        for role in ("annotator", "reviewer", "admin", None):
            for requested in sorted(OBJECT_STATES):
                assert state_for(None, requested, role, "review") in OBJECT_STATES

    def test_an_unknown_verb_leaves_the_object_where_it_was(self):
        # A geometry edit carries `adjust_geometry`, which is not a verdict and must not move the state.
        assert state_for("adjust_geometry", None, "reviewer", "review") == "review"

    def test_nothing_to_say_means_nothing_changes(self):
        assert state_for(None, None, "reviewer", None) is None

    def test_the_clamp_is_reportable(self):
        # A caller that asked for one thing and got another should be able to say so rather than surprise
        # somebody with a state they did not choose.
        assert was_clamped("accept", "accepted", "annotator", "review") is True
        assert was_clamped("accept", "accepted", "reviewer", "review") is False
        assert was_clamped("reject", None, "annotator", "review") is False
