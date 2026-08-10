"""A background job that dies must say so, with enough context to know which job.

This system's defining failure is silence, not crashes. In one week: a task that died on a foreign key with
nobody to catch it, a relabel run that ended at frame 13 of 25 and said so only in a log nobody was reading,
a gold set that shrank from 400 objects to 47 and kept reporting confident numbers, and a credential that
answered 401 to every request while all its unit tests passed.

`asyncio.create_task` does not strictly lose an exception: asyncio reports an unretrieved task exception when
the task is garbage collected. But it arrives late, on the asyncio logger rather than the structured one, and
without the run id the caller had in hand at launch, so what reaches an operator is a traceback with nothing
to attach it to. These tests are about the attachment.
"""

from __future__ import annotations

import asyncio

import pytest

from core import observability
from core.observability import capture, init_observability, running_tasks, spawn, status


@pytest.fixture
def caught(monkeypatch):
    """Collect what the structured logger was told, instead of asserting on formatted output."""
    seen: list[tuple[str, dict]] = []

    class _Log:
        def info(self, event, **kw): seen.append((event, kw))
        def warning(self, event, **kw): seen.append((event, kw))
        def error(self, event, **kw): seen.append((event, kw))

    monkeypatch.setattr(observability, "log", _Log())
    return seen


async def test_a_failing_task_is_reported_with_its_context(caught):
    """The whole point: 'task.failed' naming the run, not an anonymous traceback at collection time."""
    async def boom():
        raise RuntimeError("the job died")

    t = spawn(boom(), name="relabel_all", run_id="abc-123", frames=200)
    await asyncio.sleep(0.05)

    failed = [kw for ev, kw in caught if ev == "task.failed"]
    assert failed, "a dead background job must produce an error event"
    assert failed[0]["task"] == "relabel_all"
    assert failed[0]["run_id"] == "abc-123"
    assert "RuntimeError" in failed[0]["error"] and "the job died" in failed[0]["error"]
    assert t.done()


async def test_a_successful_task_says_so_too():
    """So 'it never finished' and 'it finished and did nothing' are distinguishable."""
    async def fine():
        return 1

    spawn(fine(), name="ok-job")
    await asyncio.sleep(0.05)


async def test_a_cancelled_task_is_not_reported_as_a_failure(caught):
    """Shutdown cancels the watchdog every time. Treating that as an error would train people to ignore the
    channel that matters."""
    async def forever():
        await asyncio.sleep(30)

    t = spawn(forever(), name="watchdog")
    await asyncio.sleep(0.01)
    t.cancel()
    await asyncio.sleep(0.05)

    assert not [kw for ev, kw in caught if ev == "task.failed"]
    assert [kw for ev, kw in caught if ev == "task.cancelled"]


async def test_a_running_task_is_held_by_a_strong_reference():
    """asyncio keeps only a weak reference, so a task nothing else holds can be collected mid-flight and
    simply stop. For a corpus job that means silent partial work."""
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.sleep(0.2)

    spawn(work(), name="held-job")   # deliberately not assigned
    await started.wait()
    assert "held-job" in running_tasks()

    import gc
    gc.collect()
    assert "held-job" in running_tasks(), "a collected task would stop with no trace"


async def test_a_finished_task_stops_being_reported_as_running():
    async def quick():
        return None

    spawn(quick(), name="brief-job")
    await asyncio.sleep(0.05)
    assert "brief-job" not in running_tasks()


def test_capture_records_a_deliberately_swallowed_failure(caught):
    """Some swallows are correct: a failed notification must not roll back the work it was announcing. What
    is wrong is the failure existing only as a warning line nobody reads."""
    capture(ValueError("notify failed"), kind="model_promoted", subject="mr-real-v1")

    got = [kw for ev, kw in caught if ev == "observability.captured"]
    assert got and got[0]["kind"] == "model_promoted"
    assert "ValueError" in got[0]["error"]


def test_context_values_are_flattened_and_bounded(caught):
    """Structured loggers reject nested values, and an unbounded field turns one bad job into an unreadable
    log line."""
    capture(ValueError("x"), payload={"a": [1, 2, 3]}, long="y" * 5000)

    kw = [k for ev, k in caught if ev == "observability.captured"][0]
    assert isinstance(kw["payload"], str)
    assert len(kw["long"]) <= 200


def test_it_is_inert_without_configuration(monkeypatch):
    """No DSN and no endpoint must mean nothing is initialised, so a developer machine sends nothing
    anywhere and turning telemetry on is a deployment change."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setitem(observability._state, "sentry", False)
    monkeypatch.setitem(observability._state, "tracing", False)

    out = init_observability()
    assert out == {"sentry": False, "tracing": False}


def test_status_reports_what_is_actually_on_not_what_was_asked_for():
    s = status()
    assert set(s) >= {"sentry", "tracing", "background_tasks"}
    assert isinstance(s["background_tasks"], list)


def test_a_broken_sentry_dsn_does_not_stop_the_app_booting(monkeypatch, caught):
    """Telemetry is the least important thing in the process and must never be the reason it will not start."""
    monkeypatch.setenv("SENTRY_DSN", "this-is-not-a-dsn")
    monkeypatch.setitem(observability._state, "sentry", False)

    out = init_observability()
    assert out["sentry"] is False
    assert [kw for ev, kw in caught if ev == "observability.sentry_failed"]
