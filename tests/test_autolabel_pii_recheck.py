"""Auto-labelling gives every frame a second redaction look inside its vehicles and people.

Measured on 91 raw dashcam frames against 112 hand-checked readable plates, the whole-frame pass at ingest
blurred 92 and missed most two-line plates; the region re-check, which needs the vehicle boxes that only
auto-labelling produces, brought it to 105. These pin the call and its accounting; no model is loaded.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.config import PiiSettings
from services.autolabel import runner


def _fd():
    return SimpleNamespace(frame=SimpleNamespace(frame_id="f1", session_id="s1"))


def _run(monkeypatch, result=None, exc: Exception | None = None) -> dict:
    async def fake(db, store, anonymizer, frame, *, apply):
        assert apply is True
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr("services.anonymize.recheck.recheck_frame", fake)
    monkeypatch.setattr("services.anonymize.anonymizer.get_anonymizer", lambda: object())
    totals: dict = {}
    asyncio.run(runner._recheck_frame_pii(None, None, _fd(), totals))
    return totals


def test_on_by_default():
    assert PiiSettings().recheck_after_autolabel is True


def test_added_blurs_are_counted(monkeypatch):
    t = _run(monkeypatch, {"status": "applied", "plates_added": 2, "faces_added": 1})
    assert t["pii_recheck"] == {"frames": 1, "plates_added": 2, "faces_added": 1, "failed": 0}


def test_a_failure_is_counted_not_hidden(monkeypatch):
    t = _run(monkeypatch, exc=RuntimeError("store refused"))
    assert t["pii_recheck"]["failed"] == 1 and t["pii_recheck"]["frames"] == 0


@pytest.mark.parametrize("status", ["unreadable", "encode_failed"])
def test_a_frame_it_could_not_look_at_is_a_failure(monkeypatch, status):
    t = _run(monkeypatch, {"status": status, "plates_added": 0, "faces_added": 0})
    assert t["pii_recheck"]["failed"] == 1


def test_the_runner_calls_it_after_objects_are_persisted():
    import inspect

    src = inspect.getsource(runner)
    persisted = src.index("await persist_frame_objects(db, store, bus, fd.frame, fused)")
    recheck = src.index("await _recheck_frame_pii(db, store, fd, totals)")
    assert persisted < recheck, "the re-check needs the objects in the database to build its windows"
