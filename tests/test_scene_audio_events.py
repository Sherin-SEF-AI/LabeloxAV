"""Milestone B (scene + audio layers): adverse-condition segmentation from frame.scene, and audio-event
region segmentation from the RMS envelope. Pure, no infra."""

from __future__ import annotations

from services.intelligence.audio_events import classify_audio_source, segment_audio_events
from services.intelligence.scene_events import adverse_conditions, segment_scene_conditions

SEC = 1_000_000_000


def test_adverse_conditions_from_scene():
    assert adverse_conditions({"weather": "rain", "time_of_day": "day"}) == ["rain"]
    assert set(adverse_conditions({"weather": "fog", "time_of_day": "night"})) == {"fog", "night"}
    assert adverse_conditions({"weather": "clear", "time_of_day": "day"}) == []
    assert adverse_conditions(None) == []


def test_scene_condition_segmentation():
    frames = [
        (0, {"weather": "rain", "time_of_day": "day"}),
        (SEC, {"weather": "rain", "time_of_day": "day"}),
        (2 * SEC, {"weather": "clear", "time_of_day": "day"}),
        (3 * SEC, {"weather": "clear", "time_of_day": "night"}),
        (4 * SEC, {"weather": "clear", "time_of_day": "night"}),
    ]
    segs = segment_scene_conditions(frames)
    by_kind = {s["kind"]: (s["t_start_ns"], s["t_end_ns"]) for s in segs}
    assert by_kind["rain"] == (0, SEC)              # the rain run closed at its last frame
    assert by_kind["night"] == (3 * SEC, 4 * SEC)


def test_audio_event_segmentation_finds_the_spike():
    env = [0.05] * 20 + [0.9, 0.95, 0.8] + [0.05] * 20   # steady road noise + a transient
    ev = segment_audio_events(env, t_start_ns=0, t_step_ns=SEC // 100, z=2.5)
    assert len(ev) == 1
    assert ev[0]["kind"] == "audio_transient" and ev[0]["peak"] >= 0.9
    assert ev[0]["t_start_ns"] < ev[0]["t_end_ns"]


def test_audio_source_classifier_is_an_honest_seam():
    assert classify_audio_source({"peak": 0.9}) == "unclassified"   # no fabricated label without a model
