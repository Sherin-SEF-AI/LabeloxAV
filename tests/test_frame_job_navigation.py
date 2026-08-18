"""Stepping to the next frame silently left the job, and took the whole feature with it.

The editor pushed `/frame/{id}` with no `?job=`, and frame meta's prev/next were session-ordered. So one
press of `]` after opening a replica job did three things at once, none of them visible: blind mode ended
and the machine pre-labels reappeared, every box drawn from then on carried no `job_id` and no
`annotator_id`, and the frame itself might not even belong to the job. The agreement pass then found
nothing for those frames and reported honest zeros about work that had actually been done.

Session order is still right for somebody reviewing a drive, which is why it stays the default. It is wrong
for somebody working an assignment, and a job's frames are not necessarily contiguous in time: a job drawn
from an explorer predicate can span sessions entirely.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns
from db.models import Frame, LabelJob, LabelProject, LabelTask, OntologyClass, OntologyVersion
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.api.routers.objects import get_frame
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db


async def _session_of_frames(db, n: int):
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
    db.add(DbSession(session_id=sid, vehicle_id="NAV-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(60), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    fids = []
    for i in range(n):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                     img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        fids.append(fid)
    await db.flush()
    await db.commit()
    return sid, fids


async def _job_over(db, frame_ids: list[uuid.UUID]) -> LabelJob:
    project = LabelProject(name=f"nav-{uuid.uuid4().hex[:8]}")
    db.add(project)
    await db.flush()
    task = LabelTask(project_id=project.project_id, name="t")
    db.add(task)
    await db.flush()
    job = LabelJob(task_id=task.task_id, frame_ids=[str(f) for f in frame_ids],
                   stage="annotation", state="new")
    db.add(job)
    await db.flush()
    await db.commit()
    return job


class TestNavigatingInsideAJob:
    async def test_next_stays_within_the_job(self):
        """The job holds frames 0, 2 and 4 of the session. Pressing next on frame 0 must reach frame 2,
        not frame 1, which the annotator was never given."""
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 6)
            job = await _job_over(db, [fids[0], fids[2], fids[4]])

            meta = await get_frame(str(fids[0]), job_id=str(job.job_id), db=db)

        assert meta["next_frame_id"] == str(fids[2])
        assert meta["prev_frame_id"] is None, "the first frame of a job has nothing before it"

    async def test_the_last_frame_of_a_job_has_no_next(self):
        """It used to walk on into the session, where a drawn box would be stamped with a job that does
        not contain the frame."""
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 6)
            job = await _job_over(db, [fids[0], fids[2], fids[4]])

            meta = await get_frame(str(fids[4]), job_id=str(job.job_id), db=db)

        assert meta["next_frame_id"] is None
        assert meta["prev_frame_id"] == str(fids[2])

    async def test_a_job_whose_frames_are_not_in_capture_order_follows_the_job(self):
        """A job built from an explorer predicate is not contiguous in time, so capture order is not its
        order."""
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 6)
            job = await _job_over(db, [fids[4], fids[1], fids[3]])

            meta = await get_frame(str(fids[4]), job_id=str(job.job_id), db=db)

        assert meta["next_frame_id"] == str(fids[1]), "it followed capture time instead of the job"


class TestWithoutAJob:
    async def test_navigation_is_unchanged(self):
        """Reviewing a drive is the other half of this, and it must behave exactly as it always did."""
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 4)
            meta = await get_frame(str(fids[1]), db=db)

        assert meta["prev_frame_id"] == str(fids[0])
        assert meta["next_frame_id"] == str(fids[2])

    async def test_a_frame_outside_the_given_job_falls_back_to_the_session(self):
        """Rather than answering with nothing: the frame is real and the caller is somewhere unexpected,
        so the useful answer is the session's neighbours, not a dead end."""
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 4)
            job = await _job_over(db, [fids[0]])

            meta = await get_frame(str(fids[2]), job_id=str(job.job_id), db=db)

        assert meta["next_frame_id"] == str(fids[3])

    async def test_an_unknown_job_does_not_break_the_frame(self):
        async with get_sessionmaker()() as db:
            _sid, fids = await _session_of_frames(db, 3)
            meta = await get_frame(str(fids[0]), job_id=str(uuid.uuid4()), db=db)
        assert meta["next_frame_id"] == str(fids[1])

    async def test_a_missing_frame_is_still_a_404(self):
        from fastapi import HTTPException

        async with get_sessionmaker()() as db:
            with pytest.raises(HTTPException) as exc:
                await get_frame(str(uuid.uuid4()), db=db)
        assert exc.value.status_code == 404
