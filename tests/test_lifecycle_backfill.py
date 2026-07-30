"""The lifecycle backfill has to say how much of itself it actually knows.

Review is append-only and keeps the pre-human source, so an object a person ruled on has a recoverable
history. Nothing else does: the agent path writes no Review row and the 3D edit path writes none either. On
this corpus that leaves well under one percent derived, and a column that hides the difference between a
derived value and a default is a column that will be believed.
"""

from __future__ import annotations

from services.annotate.lifecycle_backfill import classify


def test_a_human_confirmation_is_derived():
    assert classify("accepted", "human", "confirm") == ("human_confirmed", "derived")


def test_a_human_edit_is_derived_and_not_a_confirmation():
    """Nudging a box is not the same as saying the label is right."""
    assert classify("accepted", "human", "adjust_geometry") == ("human_edited", "derived")
    assert classify("accepted", "human", "reclassify") == ("human_edited", "derived")


def test_gate_acceptance_is_machine_accepted_not_confirmed():
    """The state the badge was conflating with a human ruling. It is a real event and its own value."""
    assert classify("auto_accept", "auto_accept", None) == ("machine_accepted", "inferred")


def test_an_untouched_proposal_is_machine_proposed():
    assert classify("review", "fused", None) == ("machine_proposed", "inferred")


def test_human_source_without_a_review_row_is_not_treated_as_confirmed():
    """The agent and 3D-edit gap.

    Something wrote the row as human and left no audit trail. Calling that confirmed would invent a ruling
    nobody made, so it defaults and records that it defaulted.
    """
    assert classify("accepted", "human", None) == ("machine_proposed", "defaulted")


def test_every_answer_carries_how_it_was_reached():
    for state, source, action in (("accepted", "human", "confirm"), ("review", "fused", None),
                                  ("accepted", "human", None), ("auto_accept", "auto_accept", None)):
        _lifecycle, basis = classify(state, source, action)
        assert basis in ("derived", "inferred", "defaulted")
