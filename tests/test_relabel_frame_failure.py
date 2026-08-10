"""One bad frame must not end a corpus-wide job, and an outage must not be mistaken for one.

A frame whose image is missing from the object store raised NoSuchKey and took out a 1,000-frame batch at
frame 209: the remaining 791 frames of work were lost to one unreadable row. Sampling 400 frames found none
missing, so this is rare rather than systematic, which is exactly the case worth surviving.

The opposite failure is what makes tolerance dangerous. If the object store or the GPU goes away every frame
fails, and a loop that shrugs each one off would march through thousands of frames marking them done having
read none of them, leaving a cursor that claims the corpus was relabelled when nothing was. So a run of
consecutive failures stops the job.

These drive the real `run_relabel_all`. Only `commit_relabel` is substituted, which is the seam the loop
already calls by module name, so the loop under test is the one that ships.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns
from db.models import AgentRun, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent import relabel_agent
from services.agent.relabel_agent import MAX_CONSECUTIVE_FRAME_FAILURES, run_relabel_all

pytestmark = pytest.mark.db


class _Commit:
    """Stands in for commit_relabel, failing on the nth frames it is asked for."""

    def __init__(self, fail_on: set[int] | str):
        self.fail_on = fail_on
        self.calls = 0

    async def __call__(self, db, fid, **kw):
        i = self.calls
        self.calls += 1
        if self.fail_on == "all" or i in self.fail_on:
            raise RuntimeError("NoSuchKey: the specified key does not exist")
        return {"run_id": str(uuid.uuid4()), "frame_id": str(fid), "relabeled": 0,
                "counts": {"total": 1, "relabel_keep": 0, "relabel_review": 0}}


async def _seed(db, n: int) -> uuid.UUID:
    """n frames, each with one machine-labelled object, in their own session."""
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-FRAMEFAIL", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    for _ in range(n):
        f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/missing.jpg", width=100, height=100)
        db.add(f)
        db.add(Object(object_id=uuid.uuid4(), frame_id=f.frame_id, class_id=1,
                      bbox=[1.0, 1.0, 30.0, 30.0], conf=0.5, source="fused", state="review"))
    await db.commit()
    return sess.session_id


async def _new_run(db) -> uuid.UUID:
    rid = uuid.uuid4()
    db.add(AgentRun(run_id=rid, kind="relabel_all", status="running", scope={}, policy={},
                    counts={}, changes={}, critic={}))
    await db.commit()
    return rid


async def _false(*a, **k):
    return False


def test_the_failure_budget_is_a_run_not_a_total():
    """Twenty in a row is an outage; twenty scattered through 30,000 frames is a corpus with some gaps. The
    counter resetting on success is what tells those apart."""
    assert 5 <= MAX_CONSECUTIVE_FRAME_FAILURES <= 100


async def test_a_single_unreadable_frame_does_not_end_the_run(monkeypatch):
    """The shape of the batch that died at frame 209 of 1000."""
    commit = _Commit(fail_on={2})
    monkeypatch.setattr(relabel_agent, "commit_relabel", commit)
    monkeypatch.setattr(relabel_agent, "training_holds_gpu", _false)

    async with get_sessionmaker()() as db:
        sid = await _seed(db, 6)
        rid = await _new_run(db)

    await run_relabel_all(rid, max_frames=6, session_id=str(sid))

    async with get_sessionmaker()() as db:
        run = await db.get(AgentRun, rid)
    assert run.status == "committed", "one bad frame must not fail the run"
    assert run.counts.get("skipped_error") == 1
    assert commit.calls == 6, "every frame after the failure was still attempted"
    assert len(run.progress.get("done", [])) == 6, (
        "the failed frame is marked done, so a resume does not stop on it again")


async def test_the_frames_after_a_failure_still_count(monkeypatch):
    commit = _Commit(fail_on={0})
    monkeypatch.setattr(relabel_agent, "commit_relabel", commit)
    monkeypatch.setattr(relabel_agent, "training_holds_gpu", _false)

    async with get_sessionmaker()() as db:
        sid = await _seed(db, 5)
        rid = await _new_run(db)

    await run_relabel_all(rid, max_frames=5, session_id=str(sid))

    async with get_sessionmaker()() as db:
        run = await db.get(AgentRun, rid)
    assert run.counts.get("frames") == 4
    assert run.counts.get("skipped_error") == 1


async def test_an_outage_stops_the_run_rather_than_marking_frames_done_unread(monkeypatch):
    commit = _Commit(fail_on="all")
    monkeypatch.setattr(relabel_agent, "commit_relabel", commit)
    monkeypatch.setattr(relabel_agent, "training_holds_gpu", _false)

    n = MAX_CONSECUTIVE_FRAME_FAILURES + 10
    async with get_sessionmaker()() as db:
        sid = await _seed(db, n)
        rid = await _new_run(db)

    await run_relabel_all(rid, max_frames=n, session_id=str(sid))

    async with get_sessionmaker()() as db:
        run = await db.get(AgentRun, rid)
    assert run.status == "error"
    assert commit.calls == MAX_CONSECUTIVE_FRAME_FAILURES, (
        "it stopped at the budget instead of walking the whole corpus")
    assert len(run.progress.get("done", [])) < n, (
        "an outage must not leave a cursor claiming frames were read")


async def test_the_failure_counter_resets_on_a_success(monkeypatch):
    """Scattered gaps are survivable however many there are; only a run of them is an outage."""
    fail = {i for i in range(0, 30, 2)}   # every other frame, so never two in a row
    commit = _Commit(fail_on=fail)
    monkeypatch.setattr(relabel_agent, "commit_relabel", commit)
    monkeypatch.setattr(relabel_agent, "training_holds_gpu", _false)

    async with get_sessionmaker()() as db:
        sid = await _seed(db, 30)
        rid = await _new_run(db)

    await run_relabel_all(rid, max_frames=30, session_id=str(sid))

    async with get_sessionmaker()() as db:
        run = await db.get(AgentRun, rid)
    assert run.status == "committed"
    assert commit.calls == 30
    assert run.counts.get("skipped_error") == 15
