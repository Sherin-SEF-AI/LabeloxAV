"""Quiet hours are a hard line for the corpus auto-label run.

The child process checks the clock only between units, so a unit that ran past its estimate kept the machine
into the working day: the first drive of a run took 8 hours on a missing index and was still going at 10:00.
The parent now stops a child that is still running when quiet hours begin, frees the vision model, and waits
them out before relaunching.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "autolabel_corpus.py"


def _load():
    spec = importlib.util.spec_from_file_location("autolabel_corpus_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Waited(Exception):
    pass


def test_a_child_still_running_at_quiet_hours_is_stopped_and_the_parent_waits(monkeypatch):
    mod = _load()
    events = []

    class Child:
        def __init__(self, cmd, *a, **k):
            assert cmd[-1] == "--child"
            events.append("start")

        def wait(self, timeout=None):
            if "terminate" in events:
                return -15
            events.append(f"wait {timeout:.0f}")
            raise subprocess.TimeoutExpired("child", timeout)

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    def sleep(s):
        events.append("sleep")
        raise _Waited

    allowed = iter([0.0, 8 * 3600.0])
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(mod, "_seconds_until_allowed", lambda w: next(allowed))
    monkeypatch.setattr(mod, "_seconds_until_quiet", lambda w: 120.0)
    monkeypatch.setattr(mod, "_unload_ollama", lambda: events.append("unload"))
    monkeypatch.setattr(mod.time, "sleep", sleep)
    monkeypatch.setattr(sys, "argv", ["autolabel_corpus.py", "--weights", "w.pt", "--quiet-hours", "10:00-18:00"])

    with pytest.raises(_Waited):
        mod.main()
    assert events == ["start", "wait 120", "terminate", "unload", "sleep"]


def test_a_child_that_stops_itself_for_quiet_hours_is_relaunched_after_them(monkeypatch):
    mod = _load()
    events = []

    class Child:
        def __init__(self, cmd, *a, **k):
            events.append("start")

        def wait(self, timeout=None):
            return mod.QUIET_EXIT

    def sleep(s):
        events.append("sleep")
        raise _Waited

    allowed = iter([0.0, 8 * 3600.0])
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(mod, "_seconds_until_allowed", lambda w: next(allowed))
    monkeypatch.setattr(mod, "_seconds_until_quiet", lambda w: 3600.0)
    monkeypatch.setattr(mod.time, "sleep", sleep)
    monkeypatch.setattr(sys, "argv", ["autolabel_corpus.py", "--weights", "w.pt", "--quiet-hours", "10:00-18:00"])

    with pytest.raises(_Waited):
        mod.main()
    assert events == ["start", "sleep"]
