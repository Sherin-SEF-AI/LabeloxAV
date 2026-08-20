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


@pytest.mark.db
async def test_an_annotator_can_find_the_issues_raised_about_their_own_work():
    """The dimension issues could not be queried on.

    Every filter was a way of asking what is wrong with a given frame, job or object. None of them let the
    person who drew the label ask what had been said about it, which is why issues read as write-only from
    the side they are addressed to: a reviewer files one, the reviewer rota is notified, and the annotator
    finds out when the job comes back, if at all.
    """
    import uuid as _uuid

    from db.models import Frame, Object, User
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.labelops import issues as issue_svc

    maker = get_sessionmaker()
    mine, theirs = _uuid.uuid4(), _uuid.uuid4()
    async with maker() as db:
        for uid, name in ((mine, "mine"), (theirs, "theirs")):
            db.add(User(user_id=uid, name=f"issue-{name}-{uid.hex[:6]}", role="annotator"))
        sess = DbSession(session_id=_uuid.uuid4(), vehicle_id="ISS-01", start_ts_ns=0, end_ts_ns=1,
                         ontology_version="test")
        db.add(sess)
        fid = _uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                     img_uri="s3://x/y.jpg", width=640, height=480))
        await db.flush()
        my_obj, their_obj = _uuid.uuid4(), _uuid.uuid4()
        db.add(Object(object_id=my_obj, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9], conf=0.5,
                      source="human", state="accepted", annotator_id=mine))
        db.add(Object(object_id=their_obj, frame_id=fid, class_id=1, bbox=[2, 2, 8, 8], conf=0.5,
                      source="human", state="accepted", annotator_id=theirs))
        await db.commit()

        await issue_svc.create_issue(db, kind="wrong_class", body="this one is mine",
                                     object_id=str(my_obj), frame_id=str(fid))
        await issue_svc.create_issue(db, kind="wrong_class", body="this one is not",
                                     object_id=str(their_obj), frame_id=str(fid))

        for_me = {i["object_id"] for i in await issue_svc.list_issues(db, about_user=str(mine))}
        assert str(my_obj) in for_me
        assert str(their_obj) not in for_me, (
            "the filter returned somebody else's feedback, which is worse than returning none")

        # Unfiltered still returns both, so the filter narrows rather than replacing the existing queries.
        both = {i["object_id"] for i in await issue_svc.list_issues(db, frame_id=str(fid))}
        assert {str(my_obj), str(their_obj)} <= both


@pytest.mark.db
async def test_the_annotator_whose_label_it_is_gets_told():
    """The rota notification does not reach the person the feedback is about.

    issue_opened is addressed to the reviewer ROLE, for a stated and good reason: who picks an issue up is
    a duty rota, not a property of the issue. The consequence was that the annotator who drew the label -
    the one participant the feedback is actually for - was the only one never told, and found out when the
    job came back, if at all. That is the slowest possible correction loop.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Frame, Notification, Object, User
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.labelops import issues as issue_svc

    maker = get_sessionmaker()
    drew_it, reviewer = _uuid.uuid4(), _uuid.uuid4()
    async with maker() as db:
        db.add(User(user_id=drew_it, name=f"drew-{drew_it.hex[:6]}", role="annotator"))
        db.add(User(user_id=reviewer, name=f"rev-{reviewer.hex[:6]}", role="reviewer"))
        sess = DbSession(session_id=_uuid.uuid4(), vehicle_id="ISS-02", start_ts_ns=0, end_ts_ns=1,
                         ontology_version="test")
        db.add(sess)
        fid, oid = _uuid.uuid4(), _uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                     img_uri="s3://x/y.jpg", width=640, height=480))
        await db.flush()
        db.add(Object(object_id=oid, frame_id=fid, class_id=1, bbox=[1, 1, 9, 9], conf=0.5,
                      source="human", state="accepted", annotator_id=drew_it))
        await db.commit()

        await issue_svc.create_issue(db, kind="wrong_class", body="that is a scooter",
                                     object_id=str(oid), frame_id=str(fid),
                                     created_by=str(reviewer))

        addressed = (await db.execute(
            select(Notification).where(Notification.user_id == drew_it))).scalars().all()
        assert addressed, "the annotator whose label it is was not told"
        assert any(n.kind == "issue_opened" for n in addressed)

        # The rota notification is still emitted; this adds a recipient rather than redirecting the issue.
        to_role = (await db.execute(
            select(Notification).where(Notification.kind == "issue_opened",
                                       Notification.role == "reviewer"))).scalars().all()
        assert to_role, "the reviewer rota stopped being notified"
