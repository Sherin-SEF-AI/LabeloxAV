"""Milestone I: curation slices. The membership predicate is a conjunction over scene axes + class / state /
city / confidence; a missing clause is unconstrained and an empty predicate is the universal slice."""

from __future__ import annotations

from services.curation.slices import matches_predicate

_RAIN_NIGHT = {"scene": {"weather": "rain", "time_of_day": "night"}, "city": "BLR",
               "classes": ["ambulance", "car"], "states": ["accepted"], "max_conf": 0.9}


def test_empty_predicate_matches_everything():
    assert matches_predicate(_RAIN_NIGHT, {}) is True


def test_scene_axis_clause():
    assert matches_predicate(_RAIN_NIGHT, {"weather": ["rain", "fog"]}) is True
    assert matches_predicate(_RAIN_NIGHT, {"weather": ["clear"]}) is False


def test_class_clause_is_membership():
    assert matches_predicate(_RAIN_NIGHT, {"class_names": ["ambulance"]}) is True
    assert matches_predicate(_RAIN_NIGHT, {"class_names": ["truck"]}) is False


def test_clauses_are_anded():
    # rain AND ambulance both hold
    assert matches_predicate(_RAIN_NIGHT, {"weather": ["rain"], "class_names": ["ambulance"]}) is True
    # rain holds but the class clause fails -> excluded
    assert matches_predicate(_RAIN_NIGHT, {"weather": ["rain"], "class_names": ["truck"]}) is False


def test_min_conf_gate():
    assert matches_predicate(_RAIN_NIGHT, {"min_conf": 0.8}) is True
    assert matches_predicate(_RAIN_NIGHT, {"min_conf": 0.95}) is False


def test_city_and_state_clauses():
    assert matches_predicate(_RAIN_NIGHT, {"cities": ["BLR"], "states": ["accepted"]}) is True
    assert matches_predicate(_RAIN_NIGHT, {"cities": ["DEL"]}) is False
