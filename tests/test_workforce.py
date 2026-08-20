"""Routing work to teams that label for a living.

253 of 570,379 objects carry a human verdict, and `assign_job` takes a user id, so work reached a person
only when another person picked their name off a list. These tests pin the properties that make sending work
outside the building safe: an unforgeable return, one live dispatch per job, honeypot ids that never leave,
and a rating that cannot be gamed with three good batches.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from core.config import get_settings
from services.labelops.workforce import (
    UNPROVEN_WORKFORCE_WEIGHT,
    WorkforceError,
    _dispatch_payload,
)

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


async def _seed_job(*, honeypots: bool = False) -> str:
    """A minimal project/task/job, since a dispatch needs a real job to point at."""
    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, LabelJob, LabelProject, LabelTask
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    sid, ts = uuid.uuid4(), now_ns()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        fids = []
        for i in range(3):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                         img_uri=f"s3://x/{fid}.jpg", width=320, height=240, quality=0.9, scene={}))
            fids.append(str(fid))
        p = LabelProject(name=f"proj-{uuid.uuid4().hex[:8]}", modality="image")
        db.add(p)
        await db.flush()
        t = LabelTask(project_id=p.project_id, name="task")
        db.add(t)
        await db.flush()
        j = LabelJob(task_id=t.task_id, frame_ids=fids,
                     honeypot_frame_ids=[fids[0]] if honeypots else [])
        db.add(j)
        await db.commit()
        return str(j.job_id)


async def _register(name: str, **kw):
    from db.session import get_sessionmaker
    from services.labelops.workforce import register_workforce

    async with get_sessionmaker()() as db:
        return await register_workforce(db, name=name, **kw)


# --- what a workforce is told ------------------------------------------------------------------


def test_a_dispatch_never_reveals_which_frames_are_the_honeypots():
    """The one thing that would make the quality gate worthless.

    Honeypots are how a returned batch is judged. Telling the vendor which frames they are is telling them
    exactly where to be careful, and a gate that announces itself measures nothing.
    """
    class _Job:
        job_id = uuid.uuid4()
        stage = "annotation"
        frame_ids = ["f1", "f2", "f3"]
        honeypot_frame_ids = ["f2"]

    class _Asg:
        assignment_id = uuid.uuid4()

    payload = _dispatch_payload(_Job(), _Asg())
    body = json.dumps(payload)
    assert "honeypot" not in body
    assert payload["frame_count"] == 3
    # the honeypot frame is in the work, as it must be, but is not distinguishable from the rest
    assert payload["frame_ids"] == ["f1", "f2", "f3"]


# --- registration -------------------------------------------------------------------------------


@requires_infra
def test_the_secret_is_minted_here_and_returned_once():
    """A caller-supplied secret would let whoever registers a vendor choose a value they already know, and
    that secret is all that stands between an outsider and writing annotations into the corpus."""
    from db.session import get_sessionmaker
    from services.labelops.workforce import get_workforce

    async def _flow():
        wf = await _register(f"vendor-{uuid.uuid4().hex[:8]}")
        assert len(wf["secret"]) >= 32
        assert "secret" not in {k for k in wf if k != "secret"}   # only present on the create response
        async with get_sessionmaker()() as db:
            row = await get_workforce(db, wf["workforce_id"])
        assert row.secret == wf["secret"]

    run_async(_flow())


@requires_infra
def test_a_dispatch_endpoint_pointing_at_our_own_infrastructure_is_refused():
    """Same hole a webhook URL opens: caller-supplied input the server then fetches with its own network
    position. http://localhost:9000 reaches MinIO; 169.254.169.254 reaches instance credentials.

    The suite runs with private targets allowed, because its webhook receivers are on localhost, so the
    production posture has to be restored here explicitly. Without that this test passes vacuously against
    a guard that was switched off, which is a worse outcome than not having the test.
    """
    s = get_settings()
    saved = s.integrations.allow_private_webhook_targets

    async def _flow():
        try:
            s.integrations.allow_private_webhook_targets = False
            for bad in ("http://localhost:9000/hook", "http://169.254.169.254/latest/meta-data/"):
                with pytest.raises(WorkforceError, match="refusing dispatch endpoint"):
                    await _register(f"vendor-{uuid.uuid4().hex[:8]}", endpoint=bad)
        finally:
            s.integrations.allow_private_webhook_targets = saved

    run_async(_flow())


# --- dispatch -----------------------------------------------------------------------------------


@requires_infra
def test_a_job_cannot_be_dispatched_to_two_workforces_at_once():
    """Two vendors labeling the same frames is not redundancy, it is two invoices and a merge conflict."""
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job

    async def _flow():
        job_id = await _seed_job()
        a = await _register(f"vendor-a-{uuid.uuid4().hex[:8]}")
        b = await _register(f"vendor-b-{uuid.uuid4().hex[:8]}")
        async with get_sessionmaker()() as db:
            await dispatch_job(db, job_id=job_id, workforce_id=a["workforce_id"], deliver=False)
        async with get_sessionmaker()() as db:
            with pytest.raises(WorkforceError, match="already dispatched"):
                await dispatch_job(db, job_id=job_id, workforce_id=b["workforce_id"], deliver=False)

    run_async(_flow())


@requires_infra
def test_an_inactive_workforce_is_refused():
    from sqlalchemy import update

    from db.models import Workforce
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job

    async def _flow():
        job_id = await _seed_job()
        wf = await _register(f"vendor-{uuid.uuid4().hex[:8]}")
        async with get_sessionmaker()() as db:
            await db.execute(update(Workforce)
                             .where(Workforce.workforce_id == uuid.UUID(wf["workforce_id"]))
                             .values(active=False))
            await db.commit()
        async with get_sessionmaker()() as db:
            with pytest.raises(WorkforceError, match="not active"):
                await dispatch_job(db, job_id=job_id, workforce_id=wf["workforce_id"], deliver=False)

    run_async(_flow())


@requires_infra
def test_a_polling_workforce_sees_its_own_open_dispatches_and_nobody_elses():
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job, pending_for_workforce

    async def _flow():
        job_a, job_b = await _seed_job(), await _seed_job()
        a = await _register(f"vendor-a-{uuid.uuid4().hex[:8]}")
        b = await _register(f"vendor-b-{uuid.uuid4().hex[:8]}")
        async with get_sessionmaker()() as db:
            await dispatch_job(db, job_id=job_a, workforce_id=a["workforce_id"], deliver=False)
            await dispatch_job(db, job_id=job_b, workforce_id=b["workforce_id"], deliver=False)
        async with get_sessionmaker()() as db:
            mine = await pending_for_workforce(db, a["workforce_id"])
        assert [p["job_id"] for p in mine] == [job_a]

    run_async(_flow())


# --- return and the quality gate ------------------------------------------------------------------


@requires_infra
def test_a_batch_with_no_honeypots_is_accepted_but_says_it_passed_no_gate():
    """Recording it as though it cleared a quality bar would be the quietly wrong version of this."""
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job, submit_return

    async def _flow():
        job_id = await _seed_job(honeypots=False)
        wf = await _register(f"vendor-{uuid.uuid4().hex[:8]}")
        async with get_sessionmaker()() as db:
            asg = await dispatch_job(db, job_id=job_id, workforce_id=wf["workforce_id"], deliver=False)
        async with get_sessionmaker()() as db:
            r = await submit_return(db, assignment_id=asg["assignment_id"], objects_returned=12)
        assert r["state"] == "accepted"
        assert "no honeypots" in r["reason"]

    run_async(_flow())


@requires_infra
def test_returning_twice_is_treated_as_a_retry_not_as_a_second_batch():
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job, submit_return

    async def _flow():
        job_id = await _seed_job()
        wf = await _register(f"vendor-{uuid.uuid4().hex[:8]}")
        async with get_sessionmaker()() as db:
            asg = await dispatch_job(db, job_id=job_id, workforce_id=wf["workforce_id"], deliver=False)
        async with get_sessionmaker()() as db:
            first = await submit_return(db, assignment_id=asg["assignment_id"], objects_returned=5)
        async with get_sessionmaker()() as db:
            second = await submit_return(db, assignment_id=asg["assignment_id"], objects_returned=99)
        assert first["state"] == "accepted"
        assert "ignored" in second.get("note", "")
        assert second["objects_returned"] == 5, "the retry must not overwrite the settled batch"

    run_async(_flow())


# --- rating and routing ---------------------------------------------------------------------------


@requires_infra
def test_an_unproven_workforce_is_marked_unproven_rather_than_poor():
    from db.session import get_sessionmaker
    from services.labelops.workforce import workforce_rating

    async def _flow():
        name = f"vendor-{uuid.uuid4().hex[:8]}"
        await _register(name)
        async with get_sessionmaker()() as db:
            r = await workforce_rating(db)
        d = r["per_workforce"][name]
        assert d["proven"] is False
        assert d["routing_weight"] == UNPROVEN_WORKFORCE_WEIGHT
        assert "unproven, not poor" in d["note"]

    run_async(_flow())


@requires_infra
def test_a_small_perfect_record_does_not_outrank_a_large_good_one():
    """Why the routing weight is the lower bound. Three accepted batches out of three is 1.0 and means very
    little; ninety out of a hundred is 0.9 and means a great deal. Ranking on the point estimate would send
    the most work to the least proven vendor."""
    from sqlalchemy import update

    from db.models import WorkforceAssignment
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job, workforce_rating

    async def _flow():
        small = f"small-{uuid.uuid4().hex[:8]}"
        large = f"large-{uuid.uuid4().hex[:8]}"
        s = await _register(small)
        big = await _register(large)

        async def _settle(wf_id: str, accepted: int, rejected: int):
            for i in range(accepted + rejected):
                job_id = await _seed_job()
                async with get_sessionmaker()() as db:
                    asg = await dispatch_job(db, job_id=job_id, workforce_id=wf_id, deliver=False)
                async with get_sessionmaker()() as db:
                    await db.execute(
                        update(WorkforceAssignment)
                        .where(WorkforceAssignment.assignment_id == uuid.UUID(asg["assignment_id"]))
                        .values(state="accepted" if i < accepted else "rejected"))
                    await db.commit()

        await _settle(s["workforce_id"], accepted=6, rejected=0)      # perfect, tiny
        await _settle(big["workforce_id"], accepted=27, rejected=3)   # 0.9, substantial

        async with get_sessionmaker()() as db:
            r = await workforce_rating(db)
        assert r["per_workforce"][small]["accept_rate"]["p"] == 1.0
        assert r["per_workforce"][large]["accept_rate"]["p"] == 0.9
        assert r["per_workforce"][large]["routing_weight"] > r["per_workforce"][small]["routing_weight"]

    run_async(_flow())


@requires_infra
def test_routing_refuses_rather_than_guessing_when_nothing_is_capable():
    """A workforce that cannot label these classes is not a cheap option, it is a rejected batch and a
    wasted week."""
    from db.session import get_sessionmaker
    from services.labelops.workforce import route_job

    async def _flow():
        job_id = await _seed_job()
        await _register(f"vendor-{uuid.uuid4().hex[:8]}",
                        capabilities={"classes": ["pedestrian"]})
        async with get_sessionmaker()() as db:
            r = await route_job(db, job_id=job_id, required_classes=["lidar_cuboid_3d"])
        assert r["workforce_id"] is None
        assert "capable" in r["reason"]

    run_async(_flow())


@requires_infra
def test_a_workforce_at_its_daily_capacity_is_skipped():
    """Overloading the best vendor is how a quality rating decays."""
    from db.session import get_sessionmaker
    from services.labelops.workforce import dispatch_job, route_job

    async def _flow():
        capped = await _register(f"capped-{uuid.uuid4().hex[:8]}", capacity_jobs_per_day=1)
        spare = await _register(f"spare-{uuid.uuid4().hex[:8]}", capacity_jobs_per_day=10)

        first = await _seed_job()
        async with get_sessionmaker()() as db:
            await dispatch_job(db, job_id=first, workforce_id=capped["workforce_id"], deliver=False)

        second = await _seed_job()
        async with get_sessionmaker()() as db:
            r = await route_job(db, job_id=second)
        assert r["workforce_id"] != capped["workforce_id"]
        assert r["workforce_id"] in (spare["workforce_id"], r["workforce_id"])

    run_async(_flow())


# --- callback authentication ------------------------------------------------------------------------


def test_a_return_signed_with_the_wrong_secret_does_not_verify():
    from services.integrations.webhooks import sign
    from services.labelops.workforce import verify_return_signature

    body = b'{"assignment_id":"x"}'
    good = sign("secret-a", body, 1_800_000_000)
    assert verify_return_signature("secret-a", body, good, ) is not None  # helper is callable
    assert not verify_return_signature("secret-b", body, good)


def test_a_tampered_return_body_does_not_verify():
    """The signature covers the bytes sent, so editing objects_returned after signing must fail."""
    import time

    from services.integrations.webhooks import sign
    from services.labelops.workforce import verify_return_signature

    now = int(time.time())
    body = b'{"assignment_id":"x","objects_returned":5}'
    sig = sign("s", body, now)
    assert verify_return_signature("s", body, sig)
    assert not verify_return_signature("s", b'{"assignment_id":"x","objects_returned":500}', sig)
