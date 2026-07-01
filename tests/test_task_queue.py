"""Milestone I: task queue state machine. Forward transitions and release/send-back are allowed; skipping
states and leaving the terminal state are refused (fail-closed)."""

from __future__ import annotations

from services.tasks.queue import valid_transition


def test_forward_flow_is_allowed():
    assert valid_transition("assigned", "in_progress")
    assert valid_transition("in_progress", "submitted")
    assert valid_transition("submitted", "done")


def test_release_and_send_back_allowed():
    assert valid_transition("in_progress", "assigned")        # annotator releases the task
    assert valid_transition("submitted", "in_progress")        # reviewer sends it back


def test_cannot_skip_states():
    assert not valid_transition("assigned", "done")
    assert not valid_transition("assigned", "submitted")


def test_done_is_terminal():
    assert not valid_transition("done", "in_progress")
    assert not valid_transition("done", "assigned")


def test_unknown_status_refused():
    assert not valid_transition("bogus", "done")
