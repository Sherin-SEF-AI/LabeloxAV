"""Per-class label precision: the sampler, and the two ways the measurement can lie.

The measurement exists so remediation targets come from evidence rather than from confidence, which is the
detector's opinion of its own output. That only works if the sample is honest and the arithmetic refuses to
flatter itself, which is what these assert.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _seed(db, *, class_name="traffic_signal", n=40, source="fused", side=40.0):
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    t0, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="CP-1", start_ts_ns=t0, end_ts_ns=t0 + seconds_to_ns(5),
                     city="BLR", sensors={}, ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t0, cam_id="c", img_uri="s3://x/a.jpg",
                 width=1920, height=1080, quality=0.9))
    await db.flush()
    cid = onto.by_name(class_name).id
    for _ in range(n):
        db.add(Object(frame_id=fid, class_id=cid, bbox=[10.0, 10.0, 10.0 + side, 10.0 + side],
                      conf=0.5, source=source, state="review"))
    await db.commit()
    return cid


class TestTheSampleIsHonest:
    @pytest.mark.asyncio
    async def test_the_same_seed_draws_the_same_crops(self):
        """A remeasurement after a fix has to compare like with like. Without a fixed seed the difference
        between the two numbers is partly the sample, and nothing separates that from the fix afterwards."""
        from db.session import get_sessionmaker
        from services.labelops.class_precision import sample_class

        async with get_sessionmaker()() as db:
            cid = await _seed(db, n=60)
            a = [o.object_id for o in await sample_class(db, cid, 12, seed=0.42)]
            b = [o.object_id for o in await sample_class(db, cid, 12, seed=0.42)]
            c = [o.object_id for o in await sample_class(db, cid, 12, seed=-0.31)]
        assert a == b
        assert a != c, "a different seed must draw a different sample or the seed does nothing"

    @pytest.mark.asyncio
    async def test_it_samples_randomly_rather_than_taking_the_most_confident(self):
        """A class's most confident detections are its best case. Measuring those reports how good the
        detector is when it is surest, which is not what a remediation decision needs to know."""
        import inspect

        from services.labelops import class_precision

        src = inspect.getsource(class_precision.sample_class)
        body = src.split('"""')[2]
        assert "md5" in body, "the draw must be hash-ordered, not id- or confidence-ordered"
        assert "conf" not in body, "the sampler must not order by confidence"

    @pytest.mark.asyncio
    async def test_human_reviewed_objects_are_excluded(self):
        """An object a person already ruled on is not evidence about the machine. Including them inflates
        the rate by exactly the amount of review that has happened."""
        from db.session import get_sessionmaker
        from services.labelops.class_precision import sample_class

        async with get_sessionmaker()() as db:
            cid = await _seed(db, class_name="median_gap", n=20, source="human")
            got = await sample_class(db, cid, 50, seed=0.42)
        assert [o for o in got if o.source == "human"] == []

    @pytest.mark.asyncio
    async def test_crops_too_small_to_judge_are_dropped(self):
        """A six-pixel box is not a label a judge can rule on, and an `unsure` from an unjudgeable crop
        costs a call and tells you nothing about the class.

        Its own class, because these assertions are about what a query returns for one class and every other
        test in this file seeds objects too. Sharing `traffic_signal` made this pass alone and fail in the
        suite, against big boxes another test had left behind.
        """
        from db.session import get_sessionmaker
        from services.labelops.class_precision import sample_class

        async with get_sessionmaker()() as db:
            cid = await _seed(db, class_name="bollard", n=25, side=6.0)
            assert await sample_class(db, cid, 50, seed=0.42, min_side_px=12.0) == []
            assert await sample_class(db, cid, 50, seed=0.42, min_side_px=0.0) != []


class TestTheJudgeReplyIsReadHonestly:
    def test_a_rejection_naming_the_asked_class_is_not_a_rejection(self):
        """It happens most on `object_fallback`, where "is this the right label?" is a confusing question
        about a class meaning none of the above. The judge answers `incorrect` while its own reason confirms
        the label: "the object is a fire hydrant, which is not in the provided class list". Counting that as
        an error reports the fallback class as broken on exactly the crops where it is working."""
        from services.autolabel.ontology import get_ontology
        from services.labelops.vlm_review import parse_judge_reply

        onto = get_ontology()
        reply = {"verdict": "incorrect", "correct_class": "object_fallback", "confidence": 0.8,
                 "reason": "the object is a fire hydrant, not in the list"}
        assert parse_judge_reply(reply, onto, given_class="object_fallback")["verdict"] == "unsure"
        # A real rejection still is one.
        assert parse_judge_reply(reply, onto, given_class="pole")["verdict"] == "incorrect"

    def test_an_unparseable_reply_becomes_unsure_rather_than_being_dropped(self):
        """Dropping it shrinks the denominator, and the crops a judge garbles are not a random subset."""
        from services.autolabel.ontology import get_ontology
        from services.labelops.vlm_review import parse_judge_reply

        assert parse_judge_reply({"verdict": "probably fine"}, get_ontology())["verdict"] == "unsure"

    def test_unsure_is_never_folded_into_either_side(self):
        """A judge that abstains on the hard crops and is scored only on the easy ones reports a precision
        that flatters itself."""
        from services.labelops.vlm_review import VERDICTS

        assert "unsure" in VERDICTS


class TestTargetSelection:
    @pytest.mark.asyncio
    async def test_targets_are_chosen_by_volume_not_by_suspicion(self):
        """Picking the classes that already look bad measures a hypothesis instead of testing it, and a
        class that is large and quietly wrong is the expensive case."""
        import inspect

        from services.labelops import class_precision

        src = inspect.getsource(class_precision.class_targets)
        # The ordering clause specifically, not the whole function: mean confidence is reported in the
        # output for context, and a substring check over the whole source trips on that.
        order_by = src.split(".order_by(")[1].split("\n")[0]
        assert "func.count" in order_by, order_by
        assert "conf" not in order_by, "targets must be ranked by count, not confidence"

    @pytest.mark.asyncio
    async def test_each_class_gets_its_own_batch_id(self):
        """Two classes sharing a denominator would report one precision for both."""
        from services.labelops.class_precision import batch_id_for

        assert batch_id_for("pole") != batch_id_for("traffic_signal")
        assert batch_id_for("pole").startswith("class-precision:")


class TestItDoesNotTakeTheMachineDown:
    """A judge sweep is worth nothing compared to the training run it could take out with it."""

    def test_the_sweep_takes_the_gpu_slot(self):
        """One card, one job. Two GPU jobs at once is not a clean failure: it is an out-of-memory part way
        through a batch, which the caller counts as a failed unit rather than as contention."""
        import inspect

        from services.labelops import class_precision

        assert "gpu_slot" in inspect.getsource(class_precision.judge_class)

    def test_it_yields_to_a_live_training_job(self):
        import inspect

        from services.labelops import class_precision

        assert "training_holds_gpu" in inspect.getsource(class_precision.wait_for_headroom)

    @pytest.mark.asyncio
    async def test_free_vram_is_derived_and_actually_reads_a_number(self):
        """`gpus()` reports used and total and no free figure. A `memory_free_mb` lookup returned None on a
        box with a working card, which made the headroom guard a silent no-op that read as "no GPU here"."""
        from unittest.mock import patch

        from services.labelops import class_precision

        with patch("services.hardening.resources.gpus",
                   return_value=[{"memory_total_mb": 16303.0, "memory_used_mb": 9000.0}]):
            assert await class_precision.free_vram_mb() == pytest.approx(7303.0)

    @pytest.mark.asyncio
    async def test_no_gpu_is_not_the_same_as_no_memory(self):
        """None means the check cannot apply. Treating it as zero would stop the sweep on every CPU host."""
        from unittest.mock import patch

        from services.labelops import class_precision

        with patch("services.hardening.resources.gpus", return_value=[]):
            assert await class_precision.free_vram_mb() is None

    @pytest.mark.asyncio
    async def test_a_failed_reading_does_not_decide_the_job(self):
        from unittest.mock import patch

        from services.labelops import class_precision

        with patch("services.hardening.resources.gpus", side_effect=RuntimeError("nvidia-smi gone")):
            assert await class_precision.free_vram_mb() is None

    def test_work_is_batched_rather_than_run_as_one_block(self):
        """So a training job that starts mid-class waits seconds, not the length of the class."""
        import inspect

        from services.labelops import class_precision

        src = inspect.getsource(class_precision.judge_class)
        assert "for i in range(0, len(objects)" in src
        assert class_precision.BATCH <= 50, "a batch this large defeats the point of re-checking between"


class TestAFailedCallIsNotAVerdict:
    def test_a_judge_that_was_never_reached_records_nothing(self):
        """`unsure` says the judge looked and declined; a failed call says it was never asked. Collapsing
        the two let an outage read as a finding: 80 of 80 `rider` crops came back "unsure" when in fact
        every call had errored."""
        import inspect

        from services.labelops import vlm_review

        assert "return None" in inspect.getsource(vlm_review._ask)
        loop = inspect.getsource(vlm_review.judge_objects)
        assert "if reply is None:" in loop and "failed += 1" in loop

    def test_it_stops_rather_than_judging_the_corpus_against_a_dead_judge(self):
        import inspect

        from services.labelops import vlm_review

        assert vlm_review.MAX_CONSECUTIVE_FAILURES <= 25
        assert "consecutive_failures" in inspect.getsource(vlm_review.judge_objects)
