"""Two annotators' boxes on one frame were one undifferentiated pile.

`Object` recorded what a label says and never who made it or which job it came from, so nothing could
compare two people's work. Everything else was already here: `services/quality/iaa.py` has the matching and
the statistics, `Issue` has the escalation, `LabelJob` has the work.

Three decisions are load-bearing and each has a test here.

Replica jobs are **blind**: they hide existing labels. 82.6% of this corpus is pre-labelled, and two people
correcting the same machine proposals are two editors of one label set, not two label sets. Agreement over
that measures how well two people agree with a third party neither can see.

Disagreements become **Issues**, not a state change. 519,550 objects are already in `review`, so flipping
state would hide a disagreement among them and would mutate the column every export selects on.

**No winner is chosen.** With 702 human-verified objects in the whole corpus, an automatic majority would be
two unverified opinions outvoting a third.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import (
    Frame,
    Issue,
    JobAgreement,
    LabelJob,
    LabelProject,
    LabelTask,
    Object,
    OntologyClass,
    OntologyVersion,
    User,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.labelops import agreement as ag
from services.labelops.jobs import JobError, assign_job, create_task

pytestmark = pytest.mark.db


async def _project_with_frames(db, n_frames: int = 3):
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts = now_ns()
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="AGREE-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(10), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    fids = []
    for i in range(n_frames):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                     img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        fids.append(fid)
    project = LabelProject(name=f"agree-{uuid.uuid4().hex[:8]}")
    db.add(project)
    await db.flush()
    await db.commit()
    return project, sid, fids


async def _user(db, name: str) -> User:
    row = User(user_id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}", role="annotator")
    db.add(row)
    await db.flush()
    return row


async def _box(db, *, frame_id, job: LabelJob, annotator: User, cls: str, bbox: list[float]) -> Object:
    onto = get_ontology()
    obj = Object(object_id=uuid.uuid4(), frame_id=frame_id, class_id=onto.by_name(cls).id, bbox=bbox,
                 conf=1.0, source="human", state="accepted", attrs={}, provenance={}, version=1,
                 job_id=job.job_id, annotator_id=annotator.user_id)
    db.add(obj)
    await db.flush()
    return obj


async def _replicated_task(db, *, n_frames: int = 2, replicas: int = 2):
    project, _sid, fids = await _project_with_frames(db, n_frames)
    out = await create_task(db, project_id=str(project.project_id), name="t",
                            predicate={"frame_ids": [str(f) for f in fids]}, replicas=replicas)
    jobs = list((await db.execute(
        select(LabelJob).where(LabelJob.task_id == uuid.UUID(out["task_id"]))
        .order_by(LabelJob.replica_index))).scalars().all())
    return out, jobs, fids


class TestCreatingReplicas:
    async def test_two_replicas_are_two_jobs_over_the_same_frames(self):
        async with get_sessionmaker()() as db:
            out, jobs, fids = await _replicated_task(db)

        assert out["replicas"] == 2
        assert len(jobs) == 2
        assert {str(f) for f in jobs[0].frame_ids} == {str(f) for f in jobs[1].frame_ids}

    async def test_they_carry_a_group_that_survives_their_frame_lists_diverging(self):
        """The pairing cannot be recovered by comparing frame_ids: seed_honeypots appends gold frames
        chosen from random.Random(job_id), so two replicas diverge before creation returns."""
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db)

        assert jobs[0].replica_group is not None
        assert jobs[0].replica_group == jobs[1].replica_group
        assert {j.replica_index for j in jobs} == {0, 1}

    async def test_an_ordinary_task_has_no_group_and_is_unchanged(self):
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db, replicas=1)
        assert len(jobs) == 1
        assert jobs[0].replica_group is None and jobs[0].replica_index == 0

    async def test_it_refuses_an_absurd_replica_count(self):
        async with get_sessionmaker()() as db:
            project, _sid, fids = await _project_with_frames(db, 1)
            with pytest.raises(JobError, match="replicas"):
                await create_task(db, project_id=str(project.project_id), name="t",
                                  predicate={"frame_ids": [str(f) for f in fids]}, replicas=99)


class TestAssignment:
    async def test_one_person_cannot_hold_both_replicas(self):
        """Agreement between one person and themselves is not a measurement, and it would read as
        perfect."""
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db)
            asha = await _user(db, "asha")
            await db.commit()
            await assign_job(db, str(jobs[0].job_id), str(asha.user_id))
            with pytest.raises(JobError, match="already holds the other replica"):
                await assign_job(db, str(jobs[1].job_id), str(asha.user_id))

    async def test_two_people_is_the_normal_case(self):
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db)
            asha, ravi = await _user(db, "asha"), await _user(db, "ravi")
            await db.commit()
            await assign_job(db, str(jobs[0].job_id), str(asha.user_id))
            await assign_job(db, str(jobs[1].job_id), str(ravi.user_id))
        assert True   # no refusal


class TestScoringAGroup:
    async def _submitted_pair(self, db, *, n_frames: int = 1):
        out, jobs, fids = await _replicated_task(db, n_frames=n_frames)
        asha, ravi = await _user(db, "asha"), await _user(db, "ravi")
        for j in jobs:
            j.state = "submitted"
        await db.commit()
        return out, jobs, fids, asha, ravi

    async def test_agreeing_annotators_score_one_and_open_nothing(self):
        async with get_sessionmaker()() as db:
            _out, jobs, fids, asha, ravi = await self._submitted_pair(db)
            box = [100.0, 100.0, 200.0, 200.0]
            await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=box)
            await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="rider", bbox=box)
            await db.commit()
            res = await ag.score_replica_group(db, str(jobs[0].replica_group))

        assert res["counts"]["frames_compared"] == 1
        assert res["counts"]["disagreements"] == 0
        assert res["counts"]["issues_opened"] == 0
        assert res["frames"][0]["detection_agreement"] == 1.0
        assert res["frames"][0]["class_agreement"] == 1.0

    async def test_a_class_disagreement_opens_an_issue_on_the_object(self):
        async with get_sessionmaker()() as db:
            _out, jobs, fids, asha, ravi = await self._submitted_pair(db)
            box = [100.0, 100.0, 200.0, 200.0]
            a_obj = await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=box)
            b_obj = await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="pedestrian", bbox=box)
            await db.commit()
            res = await ag.score_replica_group(db, str(jobs[0].replica_group))

            issues = list((await db.execute(
                select(Issue).where(Issue.kind == "disagreement",
                                    Issue.frame_id == fids[0]))).scalars().all())

        assert res["counts"]["disagreements"] == 1, "one conflict, not one per label"
        assert res["frames"][0]["detection_agreement"] == 1.0, "they found the same thing"
        assert res["frames"][0]["class_agreement"] == 0.0, "and called it different things"
        # Both labels are flagged. Flagging one would pick a winner by implication, and which one it was
        # would depend on how two random job ids happened to sort.
        assert {i.object_id for i in issues} == {a_obj.object_id, b_obj.object_id}
        assert all(i.status == "open" for i in issues)

    async def test_a_box_only_one_of_them_drew_counts_against_detection_not_class(self):
        async with get_sessionmaker()() as db:
            _out, jobs, fids, asha, ravi = await self._submitted_pair(db)
            shared = [100.0, 100.0, 200.0, 200.0]
            await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=shared)
            await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="rider", bbox=shared)
            await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider",
                       bbox=[500.0, 500.0, 600.0, 600.0])
            await db.commit()
            res = await ag.score_replica_group(db, str(jobs[0].replica_group))

        f = res["frames"][0]
        assert f["detection_agreement"] < 1.0, "one of them missed a box"
        assert f["class_agreement"] == 1.0, "what they both drew, they both named the same"
        assert res["counts"]["disagreements"] == 1

    async def test_rescoring_updates_rather_than_accumulates(self):
        """A recompute must not multiply the queue somebody has to work."""
        async with get_sessionmaker()() as db:
            _out, jobs, fids, asha, ravi = await self._submitted_pair(db)
            box = [100.0, 100.0, 200.0, 200.0]
            await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=box)
            await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="pedestrian", bbox=box)
            await db.commit()
            first = await ag.score_replica_group(db, str(jobs[0].replica_group))
            # Scoped to this frame: the test database carries every other test's disagreements too.
            after_one = list((await db.execute(
                select(Issue).where(Issue.kind == "disagreement",
                                    Issue.frame_id == fids[0]))).scalars().all())

            await ag.score_replica_group(db, str(jobs[0].replica_group))
            rows = list((await db.execute(
                select(JobAgreement).where(
                    JobAgreement.replica_group == jobs[0].replica_group))).scalars().all())
            after_two = list((await db.execute(
                select(Issue).where(Issue.kind == "disagreement",
                                    Issue.frame_id == fids[0]))).scalars().all())

        assert len(rows) == 1, "one verdict per frame per pair"
        assert first["counts"]["disagreements"] == 1, "one conflict, however many labels it touches"
        assert len(after_one) == 2, "both annotators' labels are in dispute, so both are flagged"
        assert len(after_two) == len(after_one), "rescoring multiplied the queue somebody has to work"

    async def test_it_refuses_to_score_while_somebody_is_still_working(self):
        """A partial set is not a disagreement, and an Issue on a box about to be redrawn is noise."""
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db)
            with pytest.raises(ag.AgreementError, match="still open"):
                await ag.score_replica_group(db, str(jobs[0].replica_group))

    async def test_it_refuses_a_group_that_is_not_a_pair(self):
        async with get_sessionmaker()() as db:
            _out, jobs, _fids = await _replicated_task(db, replicas=1)
            with pytest.raises(ag.AgreementError, match="needs two"):
                await ag.score_replica_group(db, str(uuid.uuid4()))

    async def test_no_object_state_is_touched(self):
        """The whole point of routing through Issue: state is what every export selects on."""
        async with get_sessionmaker()() as db:
            _out, jobs, fids, asha, ravi = await self._submitted_pair(db)
            box = [100.0, 100.0, 200.0, 200.0]
            a_obj = await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=box)
            await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="pedestrian", bbox=box)
            await db.commit()
            before = (a_obj.state, a_obj.version)
            await ag.score_replica_group(db, str(jobs[0].replica_group))
            await db.refresh(a_obj)

        assert (a_obj.state, a_obj.version) == before


class TestTheBoard:
    async def test_it_reports_the_count_the_number_came_from(self):
        """A task with three compared frames and one with three hundred both produce a number between zero
        and one, and only one of them means anything."""
        async with get_sessionmaker()() as db:
            out, jobs, fids = await _replicated_task(db, n_frames=2)
            asha, ravi = await _user(db, "asha"), await _user(db, "ravi")
            for j in jobs:
                j.state = "submitted"
            box = [100.0, 100.0, 200.0, 200.0]
            await _box(db, frame_id=fids[0], job=jobs[0], annotator=asha, cls="rider", bbox=box)
            await _box(db, frame_id=fids[0], job=jobs[1], annotator=ravi, cls="pedestrian", bbox=box)
            await db.commit()
            await ag.score_replica_group(db, str(jobs[0].replica_group))
            board = await ag.group_agreement(db, out["task_id"])

        assert board["frames_compared"] == 1
        assert board["replicas"] == 2
        assert board["disagreements"] == 1
        assert board["worst_frames"][0]["n_disagreements"] == 1

    async def test_an_unscored_task_says_so_rather_than_reporting_zero(self):
        async with get_sessionmaker()() as db:
            out, _jobs, _fids = await _replicated_task(db)
            board = await ag.group_agreement(db, out["task_id"])
        assert board["frames_compared"] == 0
        assert "no agreement has been scored" in board["detail"]
