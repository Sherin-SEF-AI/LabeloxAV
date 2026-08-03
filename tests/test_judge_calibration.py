"""Measuring the judge against human work that already happened.

Measuring a judge looked like it needed somebody's afternoon, which is why it had not happened. It does not:
the corpus holds hundreds of human rulings recorded for other reasons, and read correctly they are a
labelled evaluation set that costs nothing to collect. Three things decide whether the resulting numbers
mean anything, and each of them would produce a confident, silently false answer if got wrong.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.labelops.sampling import rogan_gladen_interval


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


# --- carrying the judge's uncertainty through -----------------------------------------------------


def test_a_correction_that_leaves_the_unit_interval_says_so():
    """The failure this exists to prevent, and it fired on the real corpus.

    Rogan-Gladen is unbounded. A judge whose measured error cannot explain the observed rate produces an
    estimate above 1.0, which clipped into range reads as a confident precision of 1.0. That is a signal
    that the model does not fit, not an answer.
    """
    out = rogan_gladen_interval(0.80,
                                sens_ci={"p": 0.76, "lo": 0.6522, "hi": 0.8425},
                                spec_ci={"p": 0.80, "lo": 0.6524, "hi": 0.8950})
    assert out["clamped"] is True
    assert "more extreme than this judge's measured error can explain" in out["note"]
    assert out["hi"] == 1.0
    assert out["lo"] < 1.0, "the interval must retain a lower bound worth reading"


def test_a_well_behaved_correction_does_not_claim_to_be_clamped():
    out = rogan_gladen_interval(0.60, sens_ci={"p": 0.95, "lo": 0.92, "hi": 0.98},
                                spec_ci={"p": 0.95, "lo": 0.92, "hi": 0.98})
    assert out["clamped"] is False and out["note"] is None
    assert 0.0 <= out["lo"] <= out["p"] <= out["hi"] <= 1.0


def test_an_uncertain_judge_yields_an_uncertain_correction():
    """The reason the point version is not enough. Measured sensitivity 0.76 and specificity 0.80 put the
    estimator's denominator anywhere between 0.30 and 0.74, so the corrected rate is uncertain by roughly a
    factor of two and its midpoint hides that."""
    tight = rogan_gladen_interval(0.7, sens_ci={"p": 0.95, "lo": 0.94, "hi": 0.96},
                                  spec_ci={"p": 0.95, "lo": 0.94, "hi": 0.96})
    loose = rogan_gladen_interval(0.7, sens_ci={"p": 0.76, "lo": 0.65, "hi": 0.85},
                                  spec_ci={"p": 0.80, "lo": 0.65, "hi": 0.90})
    assert (loose["hi"] - loose["lo"]) > (tight["hi"] - tight["lo"])


def test_a_judge_carrying_no_information_is_refused_across_its_whole_interval():
    out = rogan_gladen_interval(0.8, sens_ci={"p": 0.5, "lo": 0.4, "hi": 0.5},
                                spec_ci={"p": 0.5, "lo": 0.4, "hi": 0.5})
    assert out["p"] is None and "no information" in out["note"]


# --- assembling the set from review history --------------------------------------------------------


@requires_infra
def test_the_judge_is_asked_about_the_class_the_machine_asserted():
    """The trap that would invert the whole measurement.

    For every negative, the object's current class is the human's correction. Asking "is this a motorcycle?"
    about an object a human already relabelled to motorcycle has the judge agree, the negative scores as a
    positive, and a perfectly behaved judge measures a specificity near zero.
    """
    from db.session import get_sessionmaker
    from services.labelops.judge_calibration import build_calibration_set

    async def _flow():
        async with get_sessionmaker()() as db:
            cal = await build_calibration_set(db)
        for item in cal["negatives"]:
            corrected_to = item.detail.get("corrected_to")
            assert item.asked_class != corrected_to, (
                f"asked about '{item.asked_class}' which is what the human corrected it to; the question "
                f"must be the one the machine got wrong")
            assert item.human_says_correct is False

    run_async(_flow())


@requires_infra
def test_one_track_level_correction_counts_as_one_human_decision():
    """`reclassify_track` propagates a single judgement across a whole track.

    On the production corpus that is 164 negative objects behind 44 decisions, and counting the objects
    would shrink the specificity interval by roughly 2.3x: exactly the kind of number that looks like a
    measurement and is not one.

    Seeded here rather than asserted against whatever the corpus happens to hold, because the second is a
    claim about the data and this needs to be a claim about the grouping.
    """
    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Review, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.judge_calibration import build_calibration_set

    async def _flow():
        onto = get_ontology()
        wrong = next(c.id for c in onto.classes if c.name == "e_auto")
        right = next(c.id for c in onto.classes if c.name == "motorcycle")
        sid, fid, tid, ts = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), now_ns()

        async with get_sessionmaker()() as db:
            before = await build_calibration_set(db)

            if await db.get(OntologyVersion, onto.version) is None:
                db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
                await db.flush()
                for c in onto.classes:
                    db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                         india=c.india, map_to={}))
                await db.flush()
            db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                             ontology_version=onto.version))
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                         img_uri="s3://x/f.jpg", width=320, height=240, quality=0.9, scene={}))
            db.add(Track(track_id=tid, session_id=sid, class_id=right,
                         first_ts_ns=ts, last_ts_ns=ts + seconds_to_ns(1)))
            await db.flush()
            # Eight objects on one track, all corrected by a single track-level reclassify.
            for _ in range(8):
                oid = uuid.uuid4()
                db.add(Object(object_id=oid, frame_id=fid, class_id=right, track_id=tid,
                              bbox=[1.0, 1.0, 30.0, 30.0], conf=0.6, source="fused", state="accepted",
                              attrs={}, provenance={}, version=1))
                await db.flush()
                db.add(Review(object_id=oid, reviewer="rev", user_id=None, action="reclassify_track",
                              before={"class_id": wrong}, after={"class_id": right},
                              time_spent_ms=0, ts_ns=ts))
            await db.commit()

        async with get_sessionmaker()() as db:
            after = await build_calibration_set(db)

        added_objects = after["n_negative_objects"] - before["n_negative_objects"]
        added_decisions = after["n_negative_decisions"] - before["n_negative_decisions"]
        assert added_objects == 8, added_objects
        assert added_decisions == 1, (
            f"eight objects corrected by one track-level reclassify became {added_decisions} decisions; "
            f"counting them separately would make every interval too narrow")

    run_async(_flow())


@requires_infra
def test_the_set_has_both_classes_which_is_what_makes_it_usable():
    """Sensitivity needs positives and specificity needs negatives. The corpus has 240 accepted objects and
    essentially no rejected ones, so reading only object state gives one class and no correction is
    possible. The reclassifications are the negatives."""
    from db.session import get_sessionmaker
    from services.labelops.judge_calibration import build_calibration_set

    async def _flow():
        async with get_sessionmaker()() as db:
            cal = await build_calibration_set(db)
        assert cal["n_positive_decisions"] > 0
        assert cal["n_negative_decisions"] > 0

    run_async(_flow())


@requires_infra
def test_stored_calibration_does_not_read_ground_truth_off_the_current_state():
    """Why this cannot reuse judge_agreement.

    judge_agreement infers the human verdict from the object's state. A reclassified object ends up
    `accepted`, so every negative in this set would be counted as a positive and the specificity would
    collapse. The ground truth lives in the recorded detail instead.
    """
    import inspect

    from services.labelops import judge_calibration

    src = inspect.getsource(judge_calibration.stored_calibration)
    assert "human_says_correct" in src
    assert "Object.state" not in src, "the current state is the human's correction, not the machine's claim"


@requires_infra
def test_precision_prefers_the_retrospective_calibration_over_state_inference():
    import inspect

    from services.labelops import vlm_review

    src = inspect.getsource(vlm_review.judged_precision)
    assert "stored_calibration" in src
    idx_cal = src.index("if calibration:")
    idx_agree = src.index('elif agreement["usable"]')
    assert idx_cal < idx_agree, "calibration must be preferred, with agreement as the fallback"
