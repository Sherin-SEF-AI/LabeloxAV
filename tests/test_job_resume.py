"""An interrupted job must be distinguishable from a live one, and resumable from where it stopped.

Background work runs as `asyncio.create_task` inside the API process, so when that process ends the task
ends with it and nothing writes the row again. The run stays `running` forever and looks exactly like work
in flight. The live corpus was carrying five of these: two stranded by an API restart an hour before this
was written, one marked running for 863 hours.

The dangerous half is not the stale row, it is the reaper. Declaring a slow job dead and letting a second
copy start on the same cursor is worse than the problem being fixed, so "a job that is merely slow is never
reaped" gets as much attention here as the reaping itself.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from db.models import AgentRun
from db.session import get_sessionmaker
from services.agent.resume import (
    INTERRUPTED,
    STALE_AFTER,
    _now,
    beat,
    claim_for_resume,
    done_set,
    list_interrupted,
    reap_interrupted,
)

pytestmark = pytest.mark.db


async def _run(db, *, status="running", age=timedelta(0), progress=None, counts=None,
               kind="error_sweep") -> uuid.UUID:
    """A run whose last sign of life was `age` ago."""
    rid = uuid.uuid4()
    db.add(AgentRun(run_id=rid, kind=kind, status=status, scope={}, policy={},
                    counts=counts or {}, progress=progress or {},
                    heartbeat_at=_now() - age))
    await db.commit()
    return rid


async def test_a_job_that_stopped_beating_is_reaped():
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER + timedelta(minutes=1))
        reaped = await reap_interrupted(db)
        assert str(rid) in {r["run_id"] for r in reaped}
        assert (await db.get(AgentRun, rid)).status == INTERRUPTED


async def test_a_job_that_is_merely_slow_is_never_reaped():
    """The failure that would be worse than the bug: two workers on one cursor."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER - timedelta(minutes=1))
        reaped = await reap_interrupted(db)
        assert str(rid) not in {r["run_id"] for r in reaped}
        assert (await db.get(AgentRun, rid)).status == "running"


@pytest.mark.parametrize("terminal", ["committed", "reverted", "error"])
async def test_a_finished_job_is_left_alone_however_old(terminal):
    async with get_sessionmaker()() as db:
        rid = await _run(db, status=terminal, age=timedelta(days=40))
        await reap_interrupted(db)
        assert (await db.get(AgentRun, rid)).status == terminal


async def test_reaping_is_idempotent():
    """Startup runs it every time, and an already-interrupted run must not be re-reported as newly dead."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2)
        first = await reap_interrupted(db)
        second = await reap_interrupted(db)
        assert str(rid) in {r["run_id"] for r in first}
        assert str(rid) not in {r["run_id"] for r in second}


async def test_a_reaped_run_says_why_instead_of_showing_no_error():
    """"No error" beside a job that stopped mid-way is the misleading answer. It failed by dying."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2)
        await reap_interrupted(db)
        assert "interrupted" in ((await db.get(AgentRun, rid)).error or "")


async def test_the_cursor_and_counts_survive_reaping():
    """Without them a resume has nothing to resume from and the run is only a tombstone."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2,
                         progress={"done": ["s1", "s2"], "total": 5}, counts={"sessions": 2})
        await reap_interrupted(db)
        run = await db.get(AgentRun, rid)
        assert run.progress["done"] == ["s1", "s2"]
        assert run.counts["sessions"] == 2


async def test_a_heartbeat_keeps_a_long_job_alive_across_the_window():
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2)
        await beat(db, rid, progress={"done": ["s1"]})
        await reap_interrupted(db)
        assert (await db.get(AgentRun, rid)).status == "running"


async def test_a_heartbeat_for_a_deleted_run_does_not_raise():
    """A background task has nobody to catch its exceptions; it should stop, not explode."""
    async with get_sessionmaker()() as db:
        await beat(db, uuid.uuid4(), progress={"done": []})


async def test_a_heartbeat_never_revives_a_finished_run():
    async with get_sessionmaker()() as db:
        rid = await _run(db, status="committed")
        await beat(db, rid, counts={"sessions": 9})
        assert (await db.get(AgentRun, rid)).status == "committed"


# ------------------------------------------------------------------------------- resuming

async def test_claiming_an_interrupted_run_returns_its_cursor():
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2, progress={"done": ["s1", "s2"], "total": 4})
        await reap_interrupted(db)
        claimed = await claim_for_resume(db, rid)
        assert claimed["resumed"] is True
        assert claimed["progress"]["done"] == ["s1", "s2"]
        assert (await db.get(AgentRun, rid)).status == "running"


async def test_claiming_clears_the_interruption_message():
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=STALE_AFTER * 2)
        await reap_interrupted(db)
        await claim_for_resume(db, rid)
        assert (await db.get(AgentRun, rid)).error is None


async def test_a_live_run_cannot_be_claimed():
    """Two workers on one cursor is the outcome this refusal exists to prevent."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, age=timedelta(seconds=1))
        out = await claim_for_resume(db, rid)
        assert out.get("error") and out["status"] == "running"
        assert (await db.get(AgentRun, rid)).status == "running"


async def test_a_committed_run_cannot_be_claimed():
    """Resuming finished work would re-apply it."""
    async with get_sessionmaker()() as db:
        rid = await _run(db, status="committed")
        out = await claim_for_resume(db, rid)
        assert out.get("error") and out["status"] == "committed"


async def test_claiming_an_unknown_run_is_none_not_a_crash():
    async with get_sessionmaker()() as db:
        assert await claim_for_resume(db, uuid.uuid4()) is None


async def test_listing_marks_a_cursorless_run_as_not_resumable():
    """A job interrupted before it recorded anything can be restarted but not resumed, and saying so is the
    difference between offering a button that works and one that silently redoes everything."""
    async with get_sessionmaker()() as db:
        with_cursor = await _run(db, age=STALE_AFTER * 2, progress={"done": ["s1"]})
        without = await _run(db, age=STALE_AFTER * 2)
        await reap_interrupted(db)
        rows = {r["run_id"]: r for r in await list_interrupted(db)}
        assert rows[str(with_cursor)]["resumable"] is True
        assert rows[str(without)]["resumable"] is False


# ------------------------------------------------------------------------------- the cursor helper

def test_done_set_reads_an_empty_cursor_as_nothing_done():
    for empty in ({}, {"done": []}, {"done": None}, None):
        assert done_set(empty) == set()


def test_done_set_tolerates_a_cursor_of_the_wrong_shape():
    """Written by an older version of a job, and it must not raise inside a background task."""
    assert done_set({"done": "s1"}) == set()


def test_done_set_normalises_to_strings():
    assert done_set({"done": [uuid.UUID(int=1)]}) == {str(uuid.UUID(int=1))}


# ---- the route's allowlist, and the dispatch that must not drift from it -------------------------------
#
# A real run in the corpus (71cad603, kind reanalyze_all) got 152 frames into a 500-frame sweep, recorded
# 1,672 findings, added 15 faces and 118 plates, then had its process stop. Pressing resume returned 409
# saying the kind "has no resume path yet". It has one: services/agent/reanalyze.py::run_reanalyze_all
# reads the prior cursor, rebuilds the done set and skips the finished frames, exactly as the two
# allowlisted kinds do. The claim in the refusal was about the route's own table, not about the runner.


def test_every_resumable_kind_has_a_runner_that_honours_a_cursor():
    """The allowlist is a claim about the runners, so it is checked against them.

    A kind in the set whose runner ignores the cursor would restart from zero under a button labelled
    "resume", silently redoing work the run had already committed.
    """
    import inspect

    from services.api.routers.agent import _RESUMABLE_KINDS

    runners = {
        "error_sweep": "services.agent.error_daemon.run_error_sweep",
        "relabel_all": "services.agent.relabel_agent.run_relabel_all",
        "reanalyze_all": "services.agent.reanalyze.run_reanalyze_all",
        "drivable_backfill": "services.perception.backfill.run_drivable_backfill",
    }
    for kind in _RESUMABLE_KINDS:
        assert kind in runners, f"{kind} is allowlisted but this test does not know its runner"
        mod_path, fn_name = runners[kind].rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[fn_name])
        src = inspect.getsource(getattr(mod, fn_name))
        assert "done_set" in src, f"{kind} is allowlisted but its runner never reads a cursor"


def test_a_kind_whose_runner_can_resume_is_not_refused_by_the_route():
    """reanalyze_all records a cursor and its runner consumes it, so the route must not refuse it.

    This is the bug the console surfaced: the 409 named a cause the route had never checked against the
    runner, and 152 frames of committed work were reported as uncontinuable.
    """
    from services.api.routers.agent import _RESUMABLE_KINDS

    assert "reanalyze_all" in _RESUMABLE_KINDS


def test_the_dispatch_covers_every_allowlisted_kind_explicitly():
    """No `else` fallthrough: adding a kind without a branch would launch the wrong relauncher.

    That failure is silent and expensive. A reanalyze run resumed as a relabel sweep would write labels
    under a run whose scope and counts describe a redaction pass, and nothing would raise.
    """
    import inspect

    from services.api.routers import agent as agent_routes

    src = inspect.getsource(agent_routes.resume_run)
    for kind in agent_routes._RESUMABLE_KINDS:
        assert f'"{kind}"' in src, f"{kind} is allowlisted but resume_run never dispatches on it"
    assert "unreachable" in src or "no relauncher" in src, (
        "resume_run should refuse an unhandled kind loudly rather than falling through to a default")


async def test_releasing_a_claim_puts_the_run_back_rather_than_stranding_it():
    """A caller that claims and then cannot launch must undo the claim.

    `claim_for_resume` flips the row to running before the job starts, so a caller that bails afterwards
    leaves a row that looks like work in flight with no process behind it. That is precisely the state
    this feature exists to clear, recreated by the code meant to clear it, and the reaper would not notice
    it for hours.
    """
    from services.agent.resume import release_claim

    async with get_sessionmaker()() as db:
        rid = await _run(db, status=INTERRUPTED, progress={"done": ["a", "b"], "total": 10},
                         counts={"frames": 2})
        claimed = await claim_for_resume(db, rid)
        assert claimed["resumed"] is True
        assert (await db.get(AgentRun, rid)).status == "running"

        assert await release_claim(db, rid, "no relauncher is wired for this kind") is True
        run = await db.get(AgentRun, rid)
        assert run.status == INTERRUPTED
        assert "no relauncher" in run.error
        # The cursor and counts survive, so it is still resumable once the reason is fixed.
        assert done_set(run.progress) == {"a", "b"}
        assert run.counts["frames"] == 2


async def test_releasing_a_run_that_is_not_running_changes_nothing():
    """Idempotent, and it must not drag a committed run backwards into interrupted."""
    from services.agent.resume import release_claim

    async with get_sessionmaker()() as db:
        rid = await _run(db, status="committed")
        assert await release_claim(db, rid, "whatever") is False
        assert (await db.get(AgentRun, rid)).status == "committed"
        assert await release_claim(db, uuid.uuid4(), "unknown run") is False
