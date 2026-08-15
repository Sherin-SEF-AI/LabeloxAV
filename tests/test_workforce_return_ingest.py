"""Work could be dispatched to an outside team and their labels had no way back in.

`dispatch_job` POSTs a batch of frame ids to a vendor and `submit_return` takes a *claimed* count of
objects. Between the two there was no path by which the vendor's annotations became rows, and a Workforce
carries an HMAC secret and nothing else: no user, no role, no API key, so it could not write through the
object API either.

The hole reached further than the missing feature. `score_honeypots` grades "human-sourced objects on the
honeypot frames that are not the gold rows". For an in-house annotator that is their work; for a vendor it
was the empty set, so the gate that decides whether to accept a batch was reading rows the vendor could not
create and a vendor's accuracy was never a measurement of the vendor.

These cover the return leg and, more importantly, its refusals: an outside party writing into this corpus
must not be able to choose where.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from core.timebase import now_ns, seconds_to_ns
from db.models import (
    AgentRun,
    Frame,
    LabelJob,
    LabelProject,
    LabelTask,
    Object,
    OntologyClass,
    OntologyVersion,
    Workforce,
    WorkforceAssignment,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.labelops import return_ingest as ri

CVAT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
{images}
</annotations>
"""

IMAGE_TEMPLATE = """  <image id="{i}" name="{name}" width="1920" height="1080">
{boxes}
  </image>"""

BOX_TEMPLATE = ('    <box label="{label}" xtl="{x1}" ytl="{y1}" xbr="{x2}" ybr="{y2}" occluded="0"></box>')


def _write_cvat(tmp: Path, images: list[tuple[str, list[tuple[str, float, float, float, float]]]]) -> Path:
    """A CVAT export naming each image the way our own exporter does: `<frame_id>.jpg`."""
    blocks = []
    for i, (name, boxes) in enumerate(images):
        box_xml = "\n".join(
            BOX_TEMPLATE.format(label=lbl, x1=x1, y1=y1, x2=x2, y2=y2) for lbl, x1, y1, x2, y2 in boxes)
        blocks.append(IMAGE_TEMPLATE.format(i=i, name=name, boxes=box_xml))
    root = tmp / "returned"
    root.mkdir(parents=True, exist_ok=True)
    (root / "annotations.xml").write_text(CVAT_TEMPLATE.format(images="\n".join(blocks)))
    return root


async def _dispatched_job(db, *, n_frames: int = 2):
    """A workforce, a job over real frames, and an open assignment: the state a return arrives into."""
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
    db.add(DbSession(session_id=sid, vehicle_id="RETURN-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(10), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    frame_ids = []
    for i in range(n_frames):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                     img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        frame_ids.append(fid)
    await db.flush()

    project = LabelProject(name=f"proj-{uuid.uuid4().hex[:8]}")
    db.add(project)
    await db.flush()
    task = LabelTask(project_id=project.project_id, name="task")
    db.add(task)
    await db.flush()
    job = LabelJob(task_id=task.task_id, stage="annotation", state="in_progress",
                   frame_ids=[str(f) for f in frame_ids], honeypot_frame_ids=[])
    db.add(job)
    await db.flush()

    wf = Workforce(name=f"vendor-{uuid.uuid4().hex[:8]}", kind="vendor", secret="s" * 20, active=True)
    db.add(wf)
    await db.flush()
    asg = WorkforceAssignment(job_id=job.job_id, workforce_id=wf.workforce_id, state="dispatched")
    db.add(asg)
    await db.commit()
    await db.refresh(asg)
    return asg, frame_ids


class TestTheReturnLeg:
    async def test_a_returned_batch_becomes_objects_on_the_dispatched_frames(self, tmp_path):
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220), ("person", 5, 5, 55, 105)]),
                (f"{frames[1]}.jpg", [("car", 30, 40, 130, 240)]),
            ])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)

        assert out["counts"]["objects_written"] == 3
        assert out["counts"]["frames_matched"] == 2

        async with get_sessionmaker()() as db:
            from sqlalchemy import select
            rows = (await db.execute(select(Object).where(Object.frame_id == frames[0]))).scalars().all()
        assert len(rows) == 2
        o = rows[0]
        assert o.source == "human", "a person made this label"
        assert o.state == "review", "the honeypot verdict has not happened yet"
        # Attribution, so a batch can be found again, scored, and taken back out.
        assert o.provenance["assignment_id"] == str(asg.assignment_id)
        assert o.provenance["workforce_id"] == str(asg.workforce_id)
        assert o.provenance["ingest_run_id"] == out["run_id"]

    async def test_the_same_upload_twice_is_one_batch(self, tmp_path):
        """A vendor retrying an upload must not double the corpus."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [(f"{frames[0]}.jpg", [("car", 10, 20, 110, 220)])])
            first = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)
            second = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)

        assert second["run_id"] == first["run_id"]
        assert "already ingested" in second["note"]

        async with get_sessionmaker()() as db:
            from sqlalchemy import func, select
            n = (await db.execute(select(func.count()).select_from(Object)
                                  .where(Object.frame_id == frames[0]))).scalar()
        assert n == 1


class TestWhatItRefuses:
    async def test_a_frame_that_was_never_dispatched_is_refused(self, tmp_path):
        """The dispatch named the frames. Anything else is an outside party choosing where to write in
        somebody else's corpus, which is the whole reason this path is guarded."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            other = uuid.uuid4()
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220)]),
                (f"{other}.jpg", [("car", 10, 20, 110, 220)]),
            ])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)

        assert out["counts"]["foreign_frames"] == 1
        assert out["counts"]["objects_written"] == 1, "only the dispatched frame's labels were written"

    async def test_a_name_that_is_not_a_frame_id_is_counted_not_guessed(self, tmp_path):
        """Matching on the name is what makes a reordered or partial return safe; guessing by position
        would attach somebody's labels to the wrong image."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [("frame_0001.jpg", [("car", 10, 20, 110, 220)])])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)

        assert out["counts"]["unresolvable_names"] == 1
        assert out["counts"]["objects_written"] == 0

    async def test_a_class_the_ontology_cannot_place_is_bucketed_and_counted(self, tmp_path):
        """Not minted, because that is how an unstorable class gets created; not dropped, because
        discarding labelling somebody was paid for is the worse failure."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [(f"{frames[0]}.jpg", [("zorblatt", 10, 20, 110, 220)])])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)

        assert out["counts"]["fallback_classes"] == 1
        assert out["counts"]["objects_written"] == 1

        async with get_sessionmaker()() as db:
            from sqlalchemy import select
            row = (await db.execute(select(Object).where(Object.frame_id == frames[0]))).scalars().first()
        assert row.provenance["external_class"] == "zorblatt", "what they called it survives the mapping"

    async def test_a_decided_assignment_cannot_be_added_to(self, tmp_path):
        """A vendor must not keep writing after its verdict."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            asg.state = "accepted"
            await db.commit()
            root = _write_cvat(tmp_path, [(f"{frames[0]}.jpg", [("car", 10, 20, 110, 220)])])
            with pytest.raises(ri.ReturnIngestError, match="decided"):
                await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)


class TestTakingABatchBackOut:
    async def test_a_rejected_batch_leaves_the_corpus_whole(self, tmp_path):
        """The honeypot verdict lands after the write, so a batch that fails its bar has to be removable
        completely: per-object undo of somebody else's work is how half a batch stays behind forever."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220), ("person", 5, 5, 55, 105)]),
            ])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)
            back = await ri.revert_ingest(db, out["run_id"])

        assert back["removed"] == 2

        async with get_sessionmaker()() as db:
            from sqlalchemy import func, select
            n = (await db.execute(select(func.count()).select_from(Object)
                                  .where(Object.frame_id == frames[0]))).scalar()
            run = await db.get(AgentRun, uuid.UUID(out["run_id"]))
        assert n == 0
        assert run.status == "reverted"

    async def test_an_object_somebody_has_since_rewritten_is_left_alone(self, tmp_path):
        """A reviewer who corrected a vendor's box owns it now. Removing it underneath them would delete a
        person's work to undo a machine's bookkeeping."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220), ("person", 5, 5, 55, 105)]),
            ])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)
            oid = uuid.UUID(next(iter((await db.get(AgentRun, uuid.UUID(out["run_id"]))).changes)))
            obj = await db.get(Object, oid)
            obj.provenance = {**(obj.provenance or {}), "ingest_run_id": "edited-by-a-reviewer"}
            await db.commit()

            back = await ri.revert_ingest(db, out["run_id"])

        assert back["removed"] == 1
        assert back["kept"] == 1


class TestSettlingTheBatch:
    """The composition the upload route performs: ingest, then settle against what actually arrived. These
    exercise the two service calls in that order rather than the HTTP layer, which adds only signature
    verification that `_authenticate` already covers."""

    async def test_the_count_recorded_is_the_one_that_arrived(self, tmp_path):
        """`objects_returned` used to be whatever the vendor asserted. A number nobody checked is not a
        measurement, and it fed the assignment record that a rating is later built from."""
        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220), ("person", 5, 5, 55, 105)]),
            ])

        async with get_sessionmaker()() as db:
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)
            from services.labelops import workforce as wf_svc
            decided = await wf_svc.submit_return(
                db, assignment_id=str(asg.assignment_id),
                objects_returned=out["counts"]["objects_written"],
                detail={"ingest": out["counts"]})

        assert decided["objects_returned"] == 2, "measured from the batch, not claimed by the sender"
        assert decided["detail"]["ingest"]["frames_matched"] == 1

    async def test_a_rejected_batch_does_not_stay_in_the_corpus(self, tmp_path):
        """A gate whose refusals stay on disk decides nothing: the labels it rejected become
        indistinguishable from the ones it accepted."""
        from sqlalchemy import func, select

        async with get_sessionmaker()() as db:
            asg, frames = await _dispatched_job(db)
            root = _write_cvat(tmp_path, [
                (f"{frames[0]}.jpg", [("car", 10, 20, 110, 220), ("person", 5, 5, 55, 105)]),
            ])
            out = await ri.ingest_return(db, assignment_id=str(asg.assignment_id), fmt="cvat", root=root)
            # What the route does when submit_return comes back rejected.
            await ri.revert_ingest(db, out["run_id"])
            n = (await db.execute(select(func.count()).select_from(Object)
                                  .where(Object.frame_id == frames[0]))).scalar()
        assert n == 0
