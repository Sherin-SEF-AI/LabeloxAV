"""Milestone G: static / dynamic separation. All-parked routes to label-once; any motion (including
'stopped', which means the object was moving and then halted) routes to per-frame; nothing observed stays
per-frame rather than assuming static."""

from __future__ import annotations

from services.temporal.static_dynamic import classify_track_motion


def test_all_parked_is_static_label_once():
    c = classify_track_motion(["parked", "parked"])
    assert c["motion"] == "static" and c["label_strategy"] == "once"


def test_any_motion_is_dynamic_per_frame():
    assert classify_track_motion(["parked", "moving"])["motion"] == "dynamic"
    assert classify_track_motion(["turning"])["label_strategy"] == "per_frame"


def test_stopped_counts_as_dynamic():
    # 'stopped' means it was moving then halted, so its box changed across the clip
    assert classify_track_motion(["stopped"])["motion"] == "dynamic"


def test_unobserved_stays_per_frame():
    assert classify_track_motion([])["motion"] == "unknown"
    assert classify_track_motion([None, None])["label_strategy"] == "per_frame"


def test_none_values_are_ignored_for_observed_states():
    # an unclassified frame does not block a label-once suggestion the annotator confirms
    assert classify_track_motion([None, "parked"])["motion"] == "static"
