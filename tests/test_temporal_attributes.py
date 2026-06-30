"""Milestone G: temporal attribute transitions. Per-frame attribute values collapse into contiguous
segments; a transition is a boundary between segments; an annotation gap breaks a run rather than bridging
it into a false transition."""

from __future__ import annotations

from services.temporal.attributes import attribute_segments


def test_signal_turning_red():
    objs = [
        {"ts_ns": 0, "attrs": {"signal_state": "green"}},
        {"ts_ns": 1, "attrs": {"signal_state": "green"}},
        {"ts_ns": 2, "attrs": {"signal_state": "red"}},
        {"ts_ns": 3, "attrs": {"signal_state": "red"}},
    ]
    segs = attribute_segments(objs, "signal_state")
    assert [(s["value"], s["t_start_ns"], s["t_end_ns"]) for s in segs] == [("green", 0, 1), ("red", 2, 3)]


def test_brake_light_toggle_is_four_segments():
    objs = [{"ts_ns": i, "attrs": {"brake": (i % 2 == 0)}} for i in range(4)]   # on, off, on, off
    assert len(attribute_segments(objs, "brake")) == 4


def test_missing_attribute_breaks_the_run():
    objs = [{"ts_ns": 0, "attrs": {"x": "a"}}, {"ts_ns": 1, "attrs": {}}, {"ts_ns": 2, "attrs": {"x": "a"}}]
    segs = attribute_segments(objs, "x")
    assert len(segs) == 2 and all(s["value"] == "a" for s in segs)   # two runs, the gap is not bridged
