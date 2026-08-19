"""Blind audits end to end: the frames stay blind, and the estimate is arithmetic somebody can check.

Two properties matter here and they are different in kind.

The first is a security property. If a prediction reaches the auditor's browser the audit is void, because
the second observer stops being independent and the estimate built on them is not conservative, it is
arbitrary. So the test asks the actual fetch handler, not the editor, and asks it the way a determined
auditor would: with the job id, with somebody else's job id, and with no job id at all.

The second is arithmetic. The fixture below places every box on a 100-pixel grid so that two boxes either
coincide exactly (IoU 1.0) or are disjoint (IoU 0.0), which makes the greedy match's answer countable by
hand rather than merely plausible. Every expected number in `TestScoring` is worked out in the comments
from the counts, not read back from the implementation.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns
from db.models import (
    BlindAudit,
    BlindAuditFrame,
    Frame,
    InferenceRun,
    LabelJob,
    LabelProject,
    LabelTask,
    ModelRegistry,
    Object,
    Prediction,
    RecaptureEstimateRow,
    User,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.verdyx.blind_audit import (
    active_audit_id,
    mark_frames_labeled,
    pooled_estimate,
    score_audit,
    seed_audit,
)

pytestmark = pytest.mark.db


def _box(i: int, j: int) -> list[float]:
    """A 100x100 cell of a grid. Two cells coincide or are disjoint; nothing in between."""
    return [100.0 * i, 100.0 * j, 100.0 * i + 100.0, 100.0 * j + 100.0]


# The fixture, and the counts it is built to produce. Positions are grid cells; `p` are the model's boxes
# and `h` the blind human's. Frames A and B land in the sparse stratum (under 5 predictions), C in
# moderate (5 to 14), which is what makes the stratified path run rather than collapsing to one cell.
#
#          model                              human                      both  model_only  human_only
#   A   (0,0) (1,0) (2,0)                  (0,0) (1,0) (9,0)               2        1           1
#   B   (0,0) (1,0)                        (0,0)                           1        1           0
#   C   (0,0)..(5,0)  [6 boxes]            (0,0) (1,0) (2,0) (6,0) (7,0)   3        3           2
#
#   sparse (A+B):    both 3, model_only 2, human_only 1
#   moderate (C):    both 3, model_only 3, human_only 2
_FRAMES = {
    "A": {"pred": [(0, 0), (1, 0), (2, 0)], "human": [(0, 0), (1, 0), (9, 0)], "stratum": "sparse"},
    "B": {"pred": [(0, 0), (1, 0)], "human": [(0, 0)], "stratum": "sparse"},
    "C": {"pred": [(i, 0) for i in range(6)],
          "human": [(0, 0), (1, 0), (2, 0), (6, 0), (7, 0)], "stratum": "moderate"},
}
# On frame C the model calls the box at (2,0) a car while the human calls it a pedestrian. Class-agnostic
# counting is unaffected (it is still one box found by both), which is exactly what separates the pooled
# question from the per-class one.
_CROSS_CLASS_CELL = (2, 0)


async def _fixture(db, *, with_human: bool = True, assignee: uuid.UUID | None = None) -> dict:
    onto = get_ontology()
    ped = onto.by_name("pedestrian").id
    car = onto.by_name("sedan").id

    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="AUDIT", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto.version)
    db.add(sess)
    mv = f"audit-test-{uuid.uuid4().hex[:8]}"
    db.add(ModelRegistry(model_version=mv, task="detection"))
    await db.flush()

    run = InferenceRun(model_version=mv, gold_id=None, status="complete", frame_count=len(_FRAMES),
                       params={}, code_sha="0" * 40)
    db.add(run)
    await db.flush()

    frames: dict[str, uuid.UUID] = {}
    for k, spec in _FRAMES.items():
        f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(),
                  cam_id="front", width=2000, height=2000, img_uri=f"s3://x/{k}.jpg")
        db.add(f)
        await db.flush()
        frames[k] = f.frame_id
        for n, cell in enumerate(spec["pred"]):
            cid = car if (k == "C" and cell == _CROSS_CLASS_CELL) else ped
            db.add(Prediction(run_id=run.run_id, frame_id=f.frame_id, class_id=cid,
                              bbox=_box(*cell), conf=0.9 - 0.01 * n))
    await db.commit()

    project = LabelProject(name=f"audit-{uuid.uuid4().hex[:6]}", modality="image")
    db.add(project)
    await db.commit()

    res = await seed_audit(db, run_id=str(run.run_id), n_frames=10, score_thr=0.25,
                           project_id=str(project.project_id))
    assert "error" not in res, res
    audit_id = uuid.UUID(res["audit_id"])

    if assignee is not None and res["job_id"]:
        job = await db.get(LabelJob, uuid.UUID(res["job_id"]))
        job.assignee_id = assignee
        await db.commit()

    if with_human:
        for k, spec in _FRAMES.items():
            for cell in spec["human"]:
                db.add(Object(frame_id=frames[k], class_id=ped, bbox=_box(*cell), conf=1.0,
                              source="human", state="accepted", blind_audit_id=audit_id))
        await db.commit()

    return {"audit_id": audit_id, "run_id": run.run_id, "frames": frames, "res": res,
            "project_id": project.project_id, "ped": ped, "sedan": car}


class TestSeeding:
    async def test_frames_are_stratified_by_how_much_the_model_saw(self):
        """Two frames in sparse, one in moderate, exactly as the prediction counts imply.

        Stratifying on the model's own output is what keeps frame selection from leaking: nothing about
        what a human would find enters the choice.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db, with_human=False)
            rows = (await db.execute(select(BlindAuditFrame).where(
                BlindAuditFrame.audit_id == fx["audit_id"]))).scalars().all()
            by_frame = {r.frame_id: r.stratum for r in rows}
            assert len(rows) == 3
            for k, spec in _FRAMES.items():
                assert by_frame[fx["frames"][k]] == spec["stratum"], k
            assert fx["res"]["strata"]["sparse"]["sampled"] == 2
            assert fx["res"]["strata"]["moderate"]["sampled"] == 1

    async def test_an_empty_frame_is_still_audited(self):
        """A frame the model fired on nothing in is where a human-only object is most likely.

        Dropping those would define the audit over the frames the model already sees, which is the same
        selection bias the audit exists to remove, moved one step earlier.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db, with_human=False)
            empty = Frame(frame_id=uuid.uuid4(),
                          session_id=(await db.get(Frame, fx["frames"]["A"])).session_id,
                          ts_ns=now_ns(), cam_id="front", width=2000, height=2000, img_uri="s3://x/e.jpg")
            db.add(empty)
            await db.flush()
            # A prediction below the operating point: the run scored the frame and left nothing above it.
            db.add(Prediction(run_id=fx["run_id"], frame_id=empty.frame_id, class_id=fx["ped"],
                              bbox=_box(0, 0), conf=0.01))
            await db.commit()

            res = await seed_audit(db, run_id=str(fx["run_id"]), n_frames=10, score_thr=0.25,
                                   project_id=str(fx["project_id"]))
            picked = (await db.execute(select(BlindAuditFrame.frame_id).where(
                BlindAuditFrame.audit_id == uuid.UUID(res["audit_id"])))).scalars().all()
            assert empty.frame_id in set(picked)
            assert res["strata"]["sparse"]["available"] == 3

    async def test_a_run_that_is_not_complete_is_not_an_observer(self):
        async with get_sessionmaker()() as db:
            fx = await _fixture(db, with_human=False)
            run = await db.get(InferenceRun, fx["run_id"])
            run.status = "running"
            await db.commit()
            res = await seed_audit(db, run_id=str(fx["run_id"]), n_frames=10)
            assert "error" in res and "partial run" in res["error"]

    async def test_no_honeypots_are_seeded_into_an_audit_job(self):
        """A honeypot is a gold frame, and a gold frame arrives with its labels already drawn.

        Mixing one into an audit would put the answers in front of the auditor on the very frames the
        measurement depends on them not having seen.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db, with_human=False)
            job = await db.get(LabelJob, uuid.UUID(fx["res"]["job_id"]))
            assert job.honeypot_frame_ids in ([], None)
            assert job.replica_group is None


class TestTheFramesStayBlind:
    async def test_the_fetch_handler_returns_nothing_but_the_auditors_own_boxes(self):
        """The property the whole design rests on, asked of the server rather than the editor.

        An ordinary reviewed label is placed on an audit frame. The auditor must not receive it: they
        would be confirming somebody else's answer, and two people agreeing is not two observations.
        """
        from services.api.routers.objects import frame_objects

        async with get_sessionmaker()() as db:
            user = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            db.add(user)
            await db.commit()
            fx = await _fixture(db, assignee=user.user_id)
            frame_a = fx["frames"]["A"]
            db.add(Object(frame_id=frame_a, class_id=fx["ped"], bbox=_box(4, 4), conf=1.0,
                          source="human", state="accepted"))
            await db.commit()

            job_id = fx["res"]["job_id"]
            got = await frame_objects(str(frame_a), job_id=job_id, db=db, user=user)
            assert {o["object_id"] for o in got} == {
                str(o) for o in (await db.execute(select(Object.object_id).where(
                    Object.frame_id == frame_a,
                    Object.blind_audit_id == fx["audit_id"]))).scalars().all()}
            assert len(got) == len(_FRAMES["A"]["human"])

    async def test_dropping_the_job_id_is_not_a_way_to_see_the_answers(self):
        """The obvious defeat, and the reason the rule keys on the user as well as the job.

        A filter that only fires when the client passes a parameter is a filter the client controls.
        """
        from services.api.routers.objects import frame_objects

        async with get_sessionmaker()() as db:
            user = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            db.add(user)
            await db.commit()
            fx = await _fixture(db, assignee=user.user_id)
            frame_a = fx["frames"]["A"]
            db.add(Object(frame_id=frame_a, class_id=fx["ped"], bbox=_box(4, 4), conf=1.0,
                          source="human", state="accepted"))
            await db.commit()

            no_job = await frame_objects(str(frame_a), job_id=None, db=db, user=user)
            assert len(no_job) == len(_FRAMES["A"]["human"])
            assert all(o["bbox"] != _box(4, 4) for o in no_job)

    async def test_everybody_else_sees_the_frame_normally(self):
        """The blindness is scoped to the auditor. It is not a corpus-wide redaction.

        A reviewer looking at the same frame for any other reason must still see all of it, or seeding an
        audit would quietly hide part of the corpus from the rest of the team.
        """
        from services.api.routers.objects import frame_objects

        async with get_sessionmaker()() as db:
            auditor = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            other = User(name=f"other-{uuid.uuid4().hex[:6]}", role="reviewer")
            db.add_all([auditor, other])
            await db.commit()
            fx = await _fixture(db, assignee=auditor.user_id)
            frame_a = fx["frames"]["A"]
            db.add(Object(frame_id=frame_a, class_id=fx["ped"], bbox=_box(4, 4), conf=1.0,
                          source="human", state="accepted"))
            await db.commit()

            seen = await frame_objects(str(frame_a), job_id=None, db=db, user=other)
            assert len(seen) == len(_FRAMES["A"]["human"]) + 1
            assert any(o["bbox"] == _box(4, 4) for o in seen)

    async def test_the_frame_metadata_does_not_count_the_hidden_boxes(self):
        """The subtler leak, and the one a frontend-side hide would never have closed.

        The objects route withholds the boxes, but the frame route returns `n_objects`, which is a count of
        them. An auditor shown an empty canvas beside "14 objects" has been told exactly how many things
        they missed and how hard to keep looking, which is most of the information the hiding was for.
        """
        from services.api.routers.objects import get_frame

        async with get_sessionmaker()() as db:
            user = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            other = User(name=f"other-{uuid.uuid4().hex[:6]}", role="reviewer")
            db.add_all([user, other])
            await db.commit()
            fx = await _fixture(db, assignee=user.user_id)
            frame_a = fx["frames"]["A"]
            # Four, not three: frame A carries three audit boxes, and a tie makes the dominant-source
            # ordering arbitrary rather than wrong.
            for cell in [(4, 4), (5, 5), (6, 6), (7, 7)]:
                db.add(Object(frame_id=frame_a, class_id=fx["ped"], bbox=_box(*cell), conf=1.0,
                              source="imported", state="accepted"))
            await db.commit()

            seen = await get_frame(str(frame_a), job_id=fx["res"]["job_id"], db=db, user=user)
            assert seen["n_objects"] == len(_FRAMES["A"]["human"])
            assert seen["blind_audit_id"] == str(fx["audit_id"])
            # "mostly imported labels" is a statement about labels the auditor is not being shown.
            assert seen["annotation_source"] is None

            # Everybody else sees the real totals, including the three imported boxes.
            full = await get_frame(str(frame_a), job_id=None, db=db, user=other)
            assert full["n_objects"] == len(_FRAMES["A"]["human"]) + 4
            assert full["blind_audit_id"] is None
            assert full["annotation_source"] == "imported"

    async def test_the_blindness_lifts_once_the_audit_is_scored(self):
        # The measurement has been taken; there is nothing left to protect, and leaving the frames hidden
        # would permanently remove them from ordinary review.
        from services.api.routers.objects import frame_objects

        async with get_sessionmaker()() as db:
            user = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            db.add(user)
            await db.commit()
            fx = await _fixture(db, assignee=user.user_id)
            db.add(Object(frame_id=fx["frames"]["A"], class_id=fx["ped"], bbox=_box(4, 4), conf=1.0,
                          source="human", state="accepted"))
            await db.commit()
            await mark_frames_labeled(db, fx["audit_id"])
            await score_audit(db, str(fx["audit_id"]))

            seen = await frame_objects(str(fx["frames"]["A"]), job_id=fx["res"]["job_id"],
                                       db=db, user=user)
            assert len(seen) == len(_FRAMES["A"]["human"]) + 1

    async def test_read_and_write_resolve_the_audit_the_same_way(self):
        """One rule, both directions, or the audit silently collects nothing.

        If the fetch hid the predictions but the create failed to stamp the box, the audit would score as
        the human having found nothing, and report the model's recall as perfect exactly where it was
        being tested. That failure is invisible without this.
        """
        async with get_sessionmaker()() as db:
            user = User(name=f"auditor-{uuid.uuid4().hex[:6]}", role="annotator")
            db.add(user)
            await db.commit()
            fx = await _fixture(db, with_human=False, assignee=user.user_id)
            frame_a, job_id = fx["frames"]["A"], uuid.UUID(fx["res"]["job_id"])
            for job in (job_id, None):
                assert await active_audit_id(db, frame_id=frame_a, job_id=job,
                                             user_id=user.user_id) == fx["audit_id"]
            # A different user on the same frame is not acting under the audit, in either direction.
            assert await active_audit_id(db, frame_id=frame_a, job_id=None,
                                         user_id=uuid.uuid4()) is None


class TestScoring:
    async def test_it_refuses_while_the_frames_are_unlabelled(self):
        """An unlabelled frame is not a frame where the human found nothing.

        Scoring it as one would make the model's recall improve the less of the audit was done.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            res = await score_audit(db, str(fx["audit_id"]))
            assert "error" in res and "not a frame where the human found nothing" in res["error"]
            assert res["n_labeled"] == 0

    async def test_the_pooled_estimate_is_the_sum_of_the_strata(self):
        """Hand arithmetic from the fixture counts, none of it read back from the implementation.

          sparse    n1 = 3+2 = 5, n2 = 3+1 = 4, m2 = 3
                    N   = (6)(5)/4 - 1 = 30/4 - 1                       = 6.5
                    var = (6)(5)(5-3)(4-3) / [(4^2)(5)] = 60/80         = 0.75
          moderate  n1 = 3+3 = 6, n2 = 3+2 = 5, m2 = 3
                    N   = (7)(6)/4 - 1 = 42/4 - 1                       = 9.5
                    var = (7)(6)(6-3)(5-3) / [(4^2)(5)] = 252/80        = 3.15
          pooled    N   = 6.5 + 9.5                                     = 16.0
                    var = 0.75 + 3.15                                   = 3.9
                    model recall = (5+6) / 16 = 11/16                   = 0.6875
                    human recall = (4+5) / 16 = 9/16                    = 0.5625
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"])
            res = await score_audit(db, str(fx["audit_id"]))
            assert res["measured"] is True, res

            per = {s["stratum"]: s for s in res["per_stratum"]}
            assert abs(per["sparse"]["population"] - 6.5) < 1e-6
            assert abs(per["sparse"]["variance"] - 0.75) < 1e-6
            assert abs(per["moderate"]["population"] - 9.5) < 1e-6
            assert abs(per["moderate"]["variance"] - 3.15) < 1e-6

            pooled = res["pooled"]
            assert abs(pooled["population"] - 16.0) < 1e-6
            assert abs(pooled["variance"] - 3.9) < 1e-6
            assert abs(pooled["model_recall"] - 0.6875) < 1e-6
            assert abs(pooled["human_recall"] - 0.5625) < 1e-6
            assert (pooled["n_both"], pooled["n_model_only"], pooled["n_human_only"]) == (6, 5, 3)

    async def test_the_per_frame_counts_are_kept_not_only_the_total(self):
        """An audit whose whole human_only comes from one frame is a different finding from a spread one.

        The pooled number cannot tell those apart, so the per-frame counts are what make a surprising
        estimate openable instead of merely surprising.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"])
            await score_audit(db, str(fx["audit_id"]))
            rows = (await db.execute(select(BlindAuditFrame).where(
                BlindAuditFrame.audit_id == fx["audit_id"]))).scalars().all()
            got = {r.frame_id: (r.n_both, r.n_model_only, r.n_human_only) for r in rows}
            assert got[fx["frames"]["A"]] == (2, 1, 1)
            assert got[fx["frames"]["B"]] == (1, 1, 0)
            assert got[fx["frames"]["C"]] == (3, 3, 2)

    async def test_the_per_class_view_is_class_aware_and_does_not_sum_to_the_pooled_one(self):
        """Two questions, two matching rules, and the difference is the cross-class box on frame C.

        The model calls the box at (2,0) a car; the human calls it a pedestrian. Class-agnostically that
        is one object found by both. Class-aware it is a car the human never confirmed and a pedestrian
        the model did not find under that name.

          pedestrian  both 5, model_only 5, human_only 4
                      N = (11)(10)/6 - 1 = 110/6 - 1                    = 17.3333...
          car         both 0                                            -> unmeasurable
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"])
            res = await score_audit(db, str(fx["audit_id"]))

            assert res["per_class"][str(fx["ped"])] == [5, 5, 4]
            assert res["per_class"][str(fx["sedan"])] == [0, 1, 0]

            rows = {(r.stratum, r.class_id): r for r in (await db.execute(select(RecaptureEstimateRow)
                    .where(RecaptureEstimateRow.audit_id == fx["audit_id"]))).scalars().all()}
            ped = rows[(None, fx["ped"])]
            assert abs(ped.population - (110.0 / 6.0 - 1.0)) < 1e-3
            # A class the two observers never agreed on cannot be estimated, and says so rather than
            # returning the finite number Chapman would happily produce for it.
            car = rows[(None, fx["sedan"])]
            assert car.measured is False and "unbounded" in car.reason
            assert car.population is None

            pooled = rows[(None, None)]
            assert abs(pooled.population - 16.0) < 1e-6
            # Deliberately different from the per-class sum: they answer different questions.
            assert abs(ped.population - pooled.population) > 1.0

    async def test_every_slice_is_persisted_including_the_ones_that_could_not_conclude(self):
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"])
            await score_audit(db, str(fx["audit_id"]))
            rows = (await db.execute(select(RecaptureEstimateRow).where(
                RecaptureEstimateRow.audit_id == fx["audit_id"]))).scalars().all()
            keys = {(r.stratum, r.class_id) for r in rows}
            assert (None, None) in keys                       # pooled
            assert ("sparse", None) in keys and ("moderate", None) in keys
            assert (None, fx["ped"]) in keys and (None, fx["sedan"]) in keys
            assert all(r.run_id == fx["run_id"] for r in rows)
            assert all(r.estimator == "chapman-lp-v1" for r in rows)

    async def test_rescoring_replaces_the_estimate_rather_than_accumulating_beside_it(self):
        """Scoring is a pure function of the labels present, so a second run must not double the rows.

        The unique constraint would catch a duplicate pooled row only because it was declared NULLS NOT
        DISTINCT; under the SQL default, nulls compare unequal and two pooled rows would both be accepted.
        """
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"])
            first = await score_audit(db, str(fx["audit_id"]))
            second = await score_audit(db, str(fx["audit_id"]))
            n = (await db.execute(select(RecaptureEstimateRow).where(
                RecaptureEstimateRow.audit_id == fx["audit_id"]))).scalars().all()
            assert len(n) == 5
            assert first["pooled"]["population"] == second["pooled"]["population"]

    async def test_a_partially_labelled_audit_scores_only_what_was_labelled(self):
        """And the estimate then rests on those frames alone, which the counts make visible."""
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            await mark_frames_labeled(db, fx["audit_id"], [fx["frames"]["C"]])
            res = await score_audit(db, str(fx["audit_id"]))
            assert res["n_labeled"] == 1 and res["n_frames"] == 3
            # Frame C alone: both 3, model_only 3, human_only 2 -> the moderate row above.
            assert abs(res["pooled"]["population"] - 9.5) < 1e-6

    async def test_the_status_moves_and_the_estimate_is_readable_afterwards(self):
        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            assert (await db.get(BlindAudit, fx["audit_id"])).status == "seeded"
            await mark_frames_labeled(db, fx["audit_id"])
            assert (await db.get(BlindAudit, fx["audit_id"])).status == "labeling"
            await score_audit(db, str(fx["audit_id"]))
            audit = await db.get(BlindAudit, fx["audit_id"])
            assert audit.status == "scored" and audit.scored_at is not None

            est = await pooled_estimate(db, run_id=str(fx["run_id"]))
            assert est is not None and abs(est["population"] - 16.0) < 1e-6
            # No gold set on this run, so there is no gold recall to compare against and it says so
            # rather than substituting a number from somewhere else.
            assert est["gold_recall"] is None

    async def test_a_run_with_no_audit_has_no_estimate_rather_than_a_zero(self):
        async with get_sessionmaker()() as db:
            fx = await _fixture(db, with_human=False)
            assert await pooled_estimate(db, run_id=str(uuid.uuid4())) is None
            # Seeded but never scored is also None: nothing has been measured yet.
            assert await pooled_estimate(db, run_id=str(fx["run_id"])) is None


class TestSubmittingTheJobMarksTheFrames:
    async def test_submit_marks_every_frame_the_job_covered(self):
        from services.labelops.jobs import submit_job

        async with get_sessionmaker()() as db:
            fx = await _fixture(db)
            task = (await db.execute(select(LabelTask).where(
                LabelTask.predicate["blind_audit_id"].astext == str(fx["audit_id"])))).scalars().first()
            assert task is not None
            await submit_job(db, fx["res"]["job_id"])
            rows = (await db.execute(select(BlindAuditFrame).where(
                BlindAuditFrame.audit_id == fx["audit_id"]))).scalars().all()
            assert all(r.labeled_at is not None for r in rows)
            assert (await db.get(BlindAudit, fx["audit_id"])).status == "labeling"
