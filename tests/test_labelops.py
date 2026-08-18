"""Labeling operations: job stage/state machine, assignment concurrency, honeypot QA, issue threads.

Requires infra (DB). Everything is scoped to a throwaway session + project torn down at the end.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed_session(n_frames: int = 6):
    """A session with n_frames, each carrying one human 'gold-ish' object."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = list(onto.classes)[0].id if hasattr(onto, "classes") else 1
    sid = uuid.uuid4()
    start = now_ns()
    fids, oids = [], []
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="LABELOPS-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(n_frames), city="OPSCITY", sensors={},
                         ontology_version=onto.version))
        await db.flush()
        for i in range(n_frames):
            fid = uuid.uuid4()
            fids.append(fid)
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start + i * 1_000_000,
                         cam_id="cam_f", img_uri=f"s3://x/{fid}.jpg", width=640, height=480, quality=0.9))
            await db.flush()
            oid = uuid.uuid4()
            oids.append(oid)
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[10.0, 10.0, 50.0, 50.0],
                          conf=1.0, source="human", state="accepted", attrs={}, provenance={}))
        await db.commit()
    return str(sid), [str(f) for f in fids], [str(o) for o in oids], cid


async def _teardown(sid: str, project_id: str | None = None, gold_id: str | None = None):
    from sqlalchemy import delete

    from db.models import GoldSet, LabelProject
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        if project_id:
            await db.execute(delete(LabelProject).where(LabelProject.project_id == uuid.UUID(project_id)))
        if gold_id:
            await db.execute(delete(GoldSet).where(GoldSet.gold_id == gold_id))
        await db.execute(delete(DbSession).where(DbSession.session_id == uuid.UUID(sid)))
        await db.commit()


@requires_infra
def test_job_stage_state_machine_and_assignment_concurrency():
    from db.models import User
    from db.session import get_sessionmaker
    from services.labelops.jobs import (
        JobError,
        assign_job,
        create_project,
        create_task,
        list_jobs,
        project_board,
        set_state,
        submit_job,
    )

    async def run():
        sid, fids, oids, cid = await _seed_session(6)
        pid = None
        try:
            async with get_sessionmaker()() as db:
                u = User(name=f"ann-{uuid.uuid4().hex[:8]}", role="annotator")
                db.add(u)
                await db.commit()
                uid = str(u.user_id)

                p = await create_project(db, name=f"proj-{uuid.uuid4().hex[:6]}")
                pid = p["project_id"]

                # 6 frames split into jobs of 2 -> 3 jobs
                t = await create_task(db, project_id=pid, name="t1", session_id=sid, jobs_of=2)
                assert t["n_frames"] == 6 and t["n_jobs"] == 3, t
                job_id = t["jobs"][0]["job_id"]

                # assignment
                j = await assign_job(db, job_id, uid)
                assert j["assignee_id"] == uid
                v = j["version"]

                # optimistic concurrency: a stale version is refused, not silently applied
                with pytest.raises(JobError):
                    await assign_job(db, job_id, None, expected_version=v - 1)

                # state moves within the stage; stage is untouched
                j = await set_state(db, job_id, "in_progress")
                assert j["state"] == "in_progress" and j["stage"] == "annotation"
                assert j["started_at"] is not None, "starting work stamps started_at"

                # submit advances the STAGE and resets the state
                j = await submit_job(db, job_id)
                assert j["stage"] == "validation", j
                assert j["state"] == "new", "a new stage starts fresh, not completed"
                assert j["honeypot_failed"] is False

                j = await submit_job(db, job_id)
                assert j["stage"] == "acceptance"
                j = await submit_job(db, job_id)
                assert j["stage"] == "acceptance" and j["state"] == "completed", "acceptance is terminal"

                # filters and the board reflect it
                mine = await list_jobs(db, assignee_id=uid)
                assert len(mine) == 1
                board = await project_board(db, pid)
                cells = {(c["stage"], c["state"]): c["count"] for c in board["cells"]}
                assert cells.get(("acceptance", "completed")) == 1, cells
                assert cells.get(("annotation", "new")) == 2, cells
        finally:
            await _teardown(sid, pid)

    run_async(run())


@requires_infra
def test_honeypot_failure_sends_the_job_back_instead_of_advancing():
    """A job that fails its own quality bar must not land in the reviewer's queue looking like passed work."""
    from db.models import GoldSet, Object
    from db.session import get_sessionmaker
    from services.labelops.jobs import create_project, create_task, submit_job

    async def run():
        sid, fids, oids, cid = await _seed_session(4)
        pid = None
        gold_id = f"gold-ops-{uuid.uuid4().hex[:8]}"
        try:
            async with get_sessionmaker()() as db:
                # Seal the seeded human objects as the gold set.
                db.add(GoldSet(gold_id=gold_id, name="ops-gold", spec={}, object_ids=oids,
                               n_objects=len(oids), n_frames=len(fids), ontology_version="test"))
                await db.commit()

                # honeypot_frac 1.0 -> every frame is hidden gold; floor 0.9
                p = await create_project(db, name=f"proj-{uuid.uuid4().hex[:6]}",
                                         honeypot_frac=0.5, min_honeypot_accuracy=0.9, gold_id=gold_id)
                pid = p["project_id"]
                t = await create_task(db, project_id=pid, name="t-hp", session_id=sid, jobs_of=4)
                assert t["honeypots_seeded"] > 0, "gold frames must be mixed into the job"

                job_id = t["jobs"][0]["job_id"]
                # The annotator produced NOTHING on the honeypot frames (the gold objects are excluded from
                # the annotator's own work), so accuracy is 0 and the job must come back.
                res = await submit_job(db, job_id)
                assert res["honeypot_accuracy"] == 0.0, res
                assert res["honeypot_failed"] is True
                assert res["state"] == "rejected", res
                assert res["stage"] == "annotation", "a failed job stays in its stage, it does not advance"

                # sanity: the gold objects really are on those frames
                n = len((await db.execute(
                    __import__("sqlalchemy").select(Object.object_id)
                    .where(Object.object_id.in_([uuid.UUID(o) for o in oids])))).scalars().all())
                assert n == len(oids)
        finally:
            await _teardown(sid, pid, gold_id)

    run_async(run())


@requires_infra
def test_issue_threads_anchor_resolve_and_reopen():
    from db.models import User
    from db.session import get_sessionmaker
    from services.labelops.issues import (
        IssueError,
        comment,
        create_issue,
        list_issues,
        resolve_issue,
    )

    async def run():
        sid, fids, oids, cid = await _seed_session(2)
        try:
            async with get_sessionmaker()() as db:
                u = User(name=f"rev-{uuid.uuid4().hex[:8]}", role="reviewer")
                db.add(u)
                await db.commit()
                uid = str(u.user_id)

                # must be anchored to something
                with pytest.raises(IssueError):
                    await create_issue(db, kind="comment", body="floating")

                i = await create_issue(db, kind="wrong_class", body="this is an autorickshaw",
                                       object_id=oids[0], frame_id=fids[0], created_by=uid)
                assert i["status"] == "open" and len(i["comments"]) == 1

                i = await comment(db, i["issue_id"], "agreed, fixing", author_id=uid)
                assert len(i["comments"]) == 2
                assert i["comments"][0]["author"] is not None, "comments carry their author name"

                # a region-anchored issue for something MISSING (no object to point at)
                i2 = await create_issue(db, kind="missing", body="pedestrian not labelled",
                                        frame_id=fids[1], region=[10, 10, 60, 60], created_by=uid)
                assert i2["region"] == [10, 10, 60, 60]

                open_on_frame = await list_issues(db, frame_id=fids[0], status="open")
                assert len(open_on_frame) == 1
                assert open_on_frame[0]["n_comments"] == 2

                done = await resolve_issue(db, i["issue_id"], resolved_by=uid)
                assert done["status"] == "resolved" and done["resolved_at"] is not None
                assert not await list_issues(db, frame_id=fids[0], status="open")

                again = await resolve_issue(db, i["issue_id"], reopen=True)
                assert again["status"] == "open" and again["resolved_at"] is None
        finally:
            await _teardown(sid)

    run_async(run())


@requires_infra
def test_scorecards_report_throughput_and_quality():
    from db.models import Review, User
    from db.session import get_sessionmaker
    from services.labelops.quality import annotator_scorecards

    async def run():
        sid, fids, oids, cid = await _seed_session(2)
        try:
            async with get_sessionmaker()() as db:
                u = User(name=f"sc-{uuid.uuid4().hex[:8]}", role="annotator")
                db.add(u)
                await db.flush()
                # three reviews with a skewed time distribution: the median must not follow the outlier
                for ms in (1000, 2000, 60000):
                    db.add(Review(object_id=uuid.UUID(oids[0]), reviewer=u.name, user_id=u.user_id,
                                  action="confirm", time_spent_ms=ms, ts_ns=now_ns()))
                await db.commit()

                cards = await annotator_scorecards(db)
                mine = [c for c in cards if c["name"] == u.name]
                assert mine, "the annotator appears on the scorecard"
                c = mine[0]
                assert c["reviews"] == 3
                assert c["median_time_ms"] == 2000, f"median resists the outlier, got {c}"
                assert c["mean_time_ms"] > c["median_time_ms"], "mean is dragged by the 60s outlier"
        finally:
            await _teardown(sid)

    run_async(run())
