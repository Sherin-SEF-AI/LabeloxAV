"""The CoC schema refuses the traces that would be worth nothing.

A causal label is only worth more than a caption if it is checkable. Every refusal here exists because the
corresponding accepted-but-empty trace would look like a real label in an export and a report, and nothing
downstream could tell the difference: a decision with no cause, a critical object nobody can point at, a
model inventing a decision name outside the closed set.

The taxonomy is Alpamayo's (arXiv 2511.00088) rather than a private one, so a trace here is comparable with
the published 700K CoC traces. The tests pin that too, because quietly renaming a decision would make this
corpus speak a dialect while still calling itself CoC.
"""

from __future__ import annotations

import pytest

from services.reasoning.coc import (
    CRITICAL_CATEGORIES,
    LATERAL_DECISIONS,
    LONGITUDINAL_DECISIONS,
    META_ACTIONS,
    QA_CHECKS,
    CocError,
    CocTrace,
    CriticalComponent,
    from_dict,
)

OBJ = "11111111-2222-3333-4444-555555555555"


def _ok_component(**over) -> CriticalComponent:
    base = {"category": "critical_object", "description": "a cow standing in the lane", "object_id": OBJ}
    return CriticalComponent(**{**base, **over})


def _ok_trace(**over) -> CocTrace:
    base = {"longitudinal": "yield_agent_right_of_way",
            "components": [_ok_component()],
            "trace": "The ego yields because a cow occupies the lane ahead."}
    return CocTrace(**{**base, **over})


# ---------------------------------------------------------------------------- the schema is the published one

def test_the_decision_vocabulary_matches_the_published_taxonomy():
    """Renaming a decision would make this a dialect that calls itself CoC."""
    assert "yield_agent_right_of_way" in LONGITUDINAL_DECISIONS
    assert "lead_obstacle_following" in LONGITUDINAL_DECISIONS
    assert "out_of_lane_nudge" in LATERAL_DECISIONS
    assert "lateral_maneuver_abort" in LATERAL_DECISIONS
    assert len(LONGITUDINAL_DECISIONS) == 7
    assert len(LATERAL_DECISIONS) == 8


def test_the_critical_categories_are_the_published_seven():
    assert len(CRITICAL_CATEGORIES) == 7
    for c in ("critical_object", "traffic_light", "yield_stop_control", "road_event",
              "lane_laneline", "routing_intent", "odd_constraint"):
        assert c in CRITICAL_CATEGORIES


def test_the_qa_checklist_is_carried_as_data():
    """The review UI should ask the four questions the published pipeline audited against, not a locally
    invented rubric."""
    names = {n for n, _ in QA_CHECKS}
    assert names == {"causal_coverage", "causal_correctness", "proximate_cause", "decision_minimality"}


# ---------------------------------------------------------------------------- what makes a trace checkable

def test_a_valid_trace_passes():
    _ok_trace().validate()


def test_a_decision_with_no_cause_is_refused():
    """The whole point is the chain. A decision with an empty cause list is an assertion."""
    with pytest.raises(CocError, match="chain of causation"):
        _ok_trace(components=[]).validate()


def test_a_trace_with_no_decision_is_refused():
    with pytest.raises(CocError, match="at least one decision channel"):
        CocTrace(components=[_ok_component()], trace="something happened").validate()


def test_an_empty_composed_trace_is_refused():
    with pytest.raises(CocError, match="composed trace"):
        _ok_trace(trace="   ").validate()


def test_a_critical_object_must_cite_something_pointable():
    """This is the line between a label and a caption. 'A cow' is a sentence; the object row for that cow is
    a label that can be counted, exported, and corrected when the box is."""
    with pytest.raises(CocError, match="object_id or track_id"):
        _ok_trace(components=[_ok_component(object_id=None)]).validate()


def test_a_non_object_component_needs_no_citation():
    """Rain has no bounding box. Requiring one would push the model to invent a citation."""
    t = _ok_trace(components=[CriticalComponent(category="odd_constraint",
                                                description="heavy rain reduces visibility")])
    t.validate()


def test_a_track_citation_is_enough():
    _ok_trace(components=[_ok_component(object_id=None, track_id=OBJ)]).validate()


@pytest.mark.parametrize("bad", ["slow_down", "brake", "yield", ""])
def test_an_invented_decision_name_is_refused_not_rounded(bad):
    """A model that answers outside the closed set has not made a closed-set choice. Mapping 'slow_down' to
    the nearest known decision would turn a refusal into a label."""
    with pytest.raises(CocError):
        CocTrace(longitudinal=bad, components=[_ok_component()], trace="x").validate()


def test_an_unknown_uncertainty_tag_is_refused():
    with pytest.raises(CocError, match="uncertainty"):
        _ok_trace(components=[_ok_component(uncertainty="maybe")]).validate()


def test_a_component_with_no_description_is_refused():
    with pytest.raises(CocError, match="explains nothing"):
        _ok_trace(components=[_ok_component(description="  ")]).validate()


# ---------------------------------------------------------------------------- ego meta-actions stay absent

def test_meta_actions_are_declared_because_they_belong_to_the_schema():
    for m in ("gentle_decelerate", "strong_decelerate", "maintain_speed", "steer_left", "sharp_steer_right"):
        assert m in META_ACTIONS


def test_a_trace_is_valid_without_a_meta_action():
    """Ego kinematics need egomotion, and this corpus has ego_speed on 6 of 36,905 frames. An absent
    meta-action is honest; a guessed one reads as measured."""
    assert _ok_trace().meta_action is None
    _ok_trace().validate()


def test_an_unknown_meta_action_is_still_refused():
    with pytest.raises(CocError, match="meta action"):
        _ok_trace(meta_action="drift").validate()


# ---------------------------------------------------------------------------- parsing what a model returns

def test_from_dict_accepts_a_well_formed_model_response():
    t = from_dict({
        "longitudinal": "Yield_Agent_Right_Of_Way",
        "lateral": None,
        "trace": "The ego yields to a cow entering the lane.",
        "components": [{"category": "Critical_Object", "description": "cow entering from the left",
                        "uncertainty": "High", "object_id": OBJ}],
    })
    assert t.longitudinal == "yield_agent_right_of_way", "case and spacing from a model must normalise"
    assert t.components[0].uncertainty == "high"
    assert t.cited_object_ids() == [OBJ]


def test_from_dict_accepts_the_alternate_key_names_a_model_reaches_for():
    t = from_dict({
        "longitudinal": "stop_for_static_constraints",
        "reasoning": "A barricade blocks the lane.",
        "critical_components": [{"category": "road_event", "description": "barricade across the lane"}],
    })
    assert t.trace.startswith("A barricade")


def test_from_dict_refuses_rather_than_returning_an_empty_trace():
    """An empty object read as a valid trace would mean 'nothing caused anything', which is a claim."""
    with pytest.raises(CocError):
        from_dict({})


@pytest.mark.parametrize("bad", [None, [], "a trace", 42])
def test_from_dict_refuses_anything_that_is_not_an_object(bad):
    with pytest.raises(CocError):
        from_dict(bad)


def test_from_dict_refuses_a_components_value_that_is_not_a_list():
    with pytest.raises(CocError, match="must be a list"):
        from_dict({"longitudinal": "set_speed_tracking", "trace": "x", "components": "a cow"})


def test_round_trip_through_to_dict_survives():
    original = _ok_trace(lateral="in_lane_nudge")
    assert from_dict(original.to_dict()).to_dict() == original.to_dict()
