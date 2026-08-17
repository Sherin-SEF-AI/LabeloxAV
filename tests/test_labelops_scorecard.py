"""Four scores, four surfaces, none of them per class, and one of them rendered nowhere at all.

`annotator_scorecards` gave throughput and honeypot accuracy per person. `workforce_rating` gave batch
accept rate per vendor and no page in `web/` has ever called it. `op_precision` scores agent operations, not
labellers. `control_sample.measured_precision` scores the gate. So "how good is this labeller at this class",
which is the question that decides what work to send whom and what to pay for it, had four partial answers
and the arithmetic between them was left to whoever was looking.

Two refusals are the point of this module and each has a test: it does not average a rate across classes,
and it does not rank on a point estimate. Three right out of three is 1.0 and means very little.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns
from db.models import (
    Frame,
    LabelJob,
    LabelProject,
    LabelTask,
    Object,
    OntologyClass,
    OntologyVersion,
    Review,
    User,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.labelops import scorecard as sc


async def _seed(db):
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="CARD-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(5), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    project = LabelProject(name=f"card-{uuid.uuid4().hex[:8]}")
    db.add(project)
    await db.flush()
    task = LabelTask(project_id=project.project_id, name="t")
    db.add(task)
    await db.flush()
    job = LabelJob(task_id=task.task_id, frame_ids=[str(fid)], stage="annotation", state="submitted")
    db.add(job)
    await db.flush()
    return fid, job


async def _user(db, name: str, role: str = "annotator") -> User:
    row = User(user_id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}", role=role)
    db.add(row)
    await db.flush()
    return row


async def _labelled(db, *, frame_id, job, annotator: User, cls: str, n: int, verdict: str,
                    judge: User) -> None:
    """n labels of one class by one annotator, each later ruled on by somebody else."""
    onto = get_ontology()
    for i in range(n):
        obj = Object(object_id=uuid.uuid4(), frame_id=frame_id, class_id=onto.by_name(cls).id,
                     bbox=[float(i), 0.0, float(i) + 10.0, 10.0], conf=1.0, source="human",
                     state="accepted", attrs={}, provenance={}, version=1,
                     job_id=job.job_id, annotator_id=annotator.user_id)
        db.add(obj)
        await db.flush()
        db.add(Review(object_id=obj.object_id, reviewer=judge.name, user_id=judge.user_id,
                      action=verdict, before=None, after=None, time_spent_ms=1000, ts_ns=now_ns()))
    await db.flush()


class TestPerClass:
    async def test_a_labeller_good_at_one_class_and_bad_at_another_is_not_averaged_away(self):
        """The mean of 0.98 and 0.2 is a number about neither class, and it is what a single accuracy
        figure would report."""
        async with get_sessionmaker()() as db:
            fid, job = await _seed(db)
            asha, judge = await _user(db, "asha"), await _user(db, "judge", "reviewer")
            await _labelled(db, frame_id=fid, job=job, annotator=asha, cls="rider", n=30,
                            verdict="confirm", judge=judge)
            await _labelled(db, frame_id=fid, job=job, annotator=asha, cls="traffic_sign", n=30,
                            verdict="reclassify", judge=judge)
            await db.commit()
            rows = await sc.people(db)

        me = next(r for r in rows if r["user_id"] == str(asha.user_id))
        by_class = {c["class_name"]: c for c in me["per_class"]}
        assert by_class["rider"]["p"] == 1.0
        assert by_class["traffic_sign"]["p"] == 0.0
        assert me["judged"] == 60
        assert 0.4 < me["accuracy"]["p"] < 0.6, "the overall figure exists, beside its classes, not instead"

    async def test_a_class_nobody_has_judged_enough_of_is_unproven_not_bad(self):
        async with get_sessionmaker()() as db:
            fid, job = await _seed(db)
            ravi, judge = await _user(db, "ravi"), await _user(db, "judge", "reviewer")
            await _labelled(db, frame_id=fid, job=job, annotator=ravi, cls="rider", n=3,
                            verdict="reclassify", judge=judge)
            await db.commit()
            rows = await sc.people(db)

        me = next(r for r in rows if r["user_id"] == str(ravi.user_id))
        cls = me["per_class"][0]
        assert cls["proven"] is False
        assert "unproven" in cls["note"]

    async def test_confirming_your_own_label_is_not_evidence_about_it(self):
        async with get_sessionmaker()() as db:
            fid, job = await _seed(db)
            solo = await _user(db, "solo")
            await _labelled(db, frame_id=fid, job=job, annotator=solo, cls="rider", n=30,
                            verdict="confirm", judge=solo)
            await db.commit()
            rows = await sc.people(db)

        me = next((r for r in rows if r["user_id"] == str(solo.user_id)), None)
        assert me is not None
        assert me["judged"] == 0, "a person endorsing themselves measured at 100%"


class TestRanking:
    async def test_three_right_out_of_three_does_not_outrank_ninety_out_of_a_hundred(self):
        """The point estimate says 1.0 beats 0.9. The lower bound says otherwise, and it is right."""
        async with get_sessionmaker()() as db:
            fid, job = await _seed(db)
            lucky, proven, judge = (await _user(db, "lucky"), await _user(db, "proven"),
                                    await _user(db, "judge", "reviewer"))
            await _labelled(db, frame_id=fid, job=job, annotator=lucky, cls="rider", n=3,
                            verdict="confirm", judge=judge)
            await _labelled(db, frame_id=fid, job=job, annotator=proven, cls="rider", n=90,
                            verdict="confirm", judge=judge)
            await _labelled(db, frame_id=fid, job=job, annotator=proven, cls="rider", n=10,
                            verdict="reclassify", judge=judge)
            await db.commit()
            rows = await sc.people(db)

        by_id = {r["user_id"]: r for r in rows}
        lucky_row, proven_row = by_id[str(lucky.user_id)], by_id[str(proven.user_id)]
        assert lucky_row["accuracy"]["p"] == 1.0 and proven_row["accuracy"]["p"] == 0.9
        assert lucky_row["accuracy"]["proven"] is False
        assert proven_row["accuracy"]["lo"] > lucky_row["accuracy"]["lo"], (
            "the unproven labeller outranked the measured one")

    async def test_an_unjudged_labeller_is_not_reported_as_perfect(self):
        async with get_sessionmaker()() as db:
            fid, job = await _seed(db)
            fresh = await _user(db, "fresh")
            obj = Object(object_id=uuid.uuid4(), frame_id=fid,
                         class_id=get_ontology().by_name("rider").id, bbox=[0.0, 0.0, 5.0, 5.0],
                         conf=1.0, source="human", state="accepted", attrs={}, provenance={},
                         version=1, job_id=job.job_id, annotator_id=fresh.user_id)
            db.add(obj)
            await db.commit()
            rows = await sc.people(db)

        me = next(r for r in rows if r["user_id"] == str(fresh.user_id))
        assert me["accuracy"]["p"] is None, "nothing checked is unknown, not perfect"
        assert me["accuracy"]["proven"] is False


class TestTheWholeAnswer:
    async def test_people_and_vendors_come_back_together(self):
        """The question is about labellers, not about employment."""
        async with get_sessionmaker()() as db:
            out = await sc.scorecards(db)
        assert "people" in out and "vendors" in out
        assert out["min_judged"] == sc.MIN_JUDGED
        assert "not a random sample" in out["caveat"]

    async def test_your_own_record_is_available_even_before_anyone_has_judged_it(self):
        async with get_sessionmaker()() as db:
            nobody = await _user(db, "nobody")
            await db.commit()
            mine = await sc.scorecard_for(db, str(nobody.user_id))
        assert mine is not None
        assert mine["judged"] == 0
        assert "has been ruled on" in mine["detail"]
        assert mine["accuracy"]["p"] is None, "an empty card is unknown, not zero"

    async def test_an_unknown_user_is_none_rather_than_an_empty_card(self):
        async with get_sessionmaker()() as db:
            assert await sc.scorecard_for(db, str(uuid.uuid4())) is None


class TestTheRoleFloors:
    """Two routes had none, found while building this."""

    def test_the_scorecards_route_is_reviewer_gated(self):
        from services.api.routers import labelops

        route = next(r for r in labelops.router.routes
                     if getattr(r, "path", "") == "/labelops/scorecards")
        assert route.dependencies, "every annotator's record was readable by anybody the API let in"

    def test_the_control_sample_routes_are_gated(self):
        """A verdict on a control sample moves the only number that says whether the auto-accept gate is
        right."""
        from services.api.routers import govern

        for path in ("/govern/control/seed", "/govern/control/{sample_id}/verdict",
                     "/govern/control/pending", "/govern/control/precision"):
            route = next(r for r in govern.router.routes if getattr(r, "path", "") == path)
            assert route.dependencies, f"{path} had no role floor"


@pytest.mark.asyncio
async def test_the_vendor_rating_is_finally_reachable():
    """workforce_rating has been computed since it was written and rendered nowhere: no page in web/ calls
    /api/workforce at all."""
    async with get_sessionmaker()() as db:
        rows = await sc.vendors(db)
    assert isinstance(rows, list)
    for r in rows:
        assert "batch" in r and "routing_weight" in r["batch"]
