"""The machine judge, and the reason its agreement rate is not precision.

253 of 570,379 objects have a human verdict. A VLM can judge all of them, and the whole risk of doing so is
that its agreement rate looks exactly like a measurement while being a blend of two things: how good the
labels are, and how good the judge is. These tests pin the separation.
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns
from services.labelops.sampling import rogan_gladen
from services.labelops.vlm_review import (
    VERDICTS,
    build_judge_prompt,
    parse_judge_reply,
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


# --- the prompt asks a different question than the autolabel path -----------------------------


def test_the_judge_is_asked_to_confirm_not_to_name():
    """The distinction the whole method rests on.

    A model asked "what is this?" always answers something, so its reply carries no information about
    whether it was sure. A model asked "the label says autorickshaw, is that right?" can decline, and the
    declines are exactly the crops worth a person's time.
    """
    p = build_judge_prompt("autorickshaw", ["e_auto", "tempo"])
    assert "labelled: autorickshaw" in p
    assert "unsure" in p and "Do not guess" in p
    assert '"verdict"' in p


def test_the_alternatives_are_offered_so_a_rejection_carries_a_correction():
    p = build_judge_prompt("autorickshaw", ["e_auto", "tempo"])
    assert "e_auto" in p and "tempo" in p


# --- parsing refuses to invent a verdict ------------------------------------------------------


class _Onto:
    def has_name(self, n):
        return n in ("e_auto", "tempo")

    def by_name(self, n):
        return type("C", (), {"id": 7 if n == "e_auto" else 9})()


def test_an_unparseable_reply_becomes_unsure_rather_than_being_dropped():
    """Dropping it would shrink the denominator, which biases the rate upward: the crops a judge garbles
    are not a random subset of the corpus."""
    for junk in ({}, {"verdict": ""}, {"verdict": "probably"}, {"verdict": None}):
        assert parse_judge_reply(junk, _Onto())["verdict"] == "unsure"


def test_a_rejection_with_a_known_class_carries_the_proposal():
    r = parse_judge_reply({"verdict": "incorrect", "correct_class": "e_auto", "confidence": 0.8}, _Onto())
    assert r["verdict"] == "incorrect" and r["proposed_class_id"] == 7 and r["confidence"] == 0.8


def test_a_proposal_outside_the_ontology_is_dropped_but_the_rejection_stands():
    """The judge disagreeing is information even when its suggested replacement is not a real class."""
    r = parse_judge_reply({"verdict": "incorrect", "correct_class": "spaceship"}, _Onto())
    assert r["verdict"] == "incorrect" and r["proposed_class_id"] is None


def test_confidence_is_clamped_and_survives_a_junk_value():
    assert parse_judge_reply({"verdict": "correct", "confidence": 5}, _Onto())["confidence"] == 1.0
    assert parse_judge_reply({"verdict": "correct", "confidence": "x"}, _Onto())["confidence"] is None


def test_a_correct_verdict_never_carries_a_proposed_class():
    """"correct" plus a replacement class is contradictory, and would relabel objects the judge agreed with."""
    r = parse_judge_reply({"verdict": "correct", "correct_class": "e_auto"}, _Onto())
    assert r["proposed_class_id"] is None


# --- correcting for the judge's own error -----------------------------------------------------


def test_a_perfect_judge_needs_no_correction():
    assert rogan_gladen(0.85, sensitivity=1.0, specificity=1.0) == pytest.approx(0.85)


def test_an_imperfect_judge_moves_the_estimate():
    """The number this exists to stop anybody quoting.

    A judge that is 90% sensitive and 80% specific, observing 85% correct, is not looking at 85% good
    labels. Reporting the raw rate embeds the judge's error in every downstream claim.
    """
    corrected = rogan_gladen(0.85, sensitivity=0.90, specificity=0.80)
    assert corrected is not None
    assert abs(corrected - 0.9286) < 1e-3, corrected
    assert corrected != 0.85


def test_a_judge_that_carries_no_information_is_refused_rather_than_answered():
    """sensitivity + specificity == 1 is a coin. The formula divides by zero either side of it and returns
    a confident-looking number, so the refusal has to be explicit."""
    assert rogan_gladen(0.85, sensitivity=0.5, specificity=0.5) is None
    assert rogan_gladen(0.85, sensitivity=0.3, specificity=0.3) is None


def test_the_corrected_rate_stays_a_rate():
    """Small-subsample noise can push the estimate outside [0, 1]; a precision of 1.04 helps nobody."""
    for p in (0.0, 0.5, 1.0):
        v = rogan_gladen(p, sensitivity=0.6, specificity=0.6)
        assert v is None or 0.0 <= v <= 1.0


# --- end to end against the database ----------------------------------------------------------


async def _seed_judgeable_batch(batch_id: str, *, tp: int = 9, fp: int = 3, fn: int = 1, tn: int = 7):
    """A batch with a stub judge's verdicts recorded and human rulings to measure them against.

    The four counts are the confusion quadrants, named for what the judge got right and wrong: tp is
    "judge said correct and the human agreed", fp is "judge said correct and the human rejected the label",
    and so on. Stated explicitly rather than derived from a rule, because the first version of this fixture
    derived them and landed on numbers where the correction happened to equal the raw rate, so a test
    asserting the correction does something passed while proving nothing.

    The defaults give sensitivity 0.9, specificity 0.7, a raw rate of 0.6 and a corrected rate of 0.5.
    """
    import cv2

    from core.storage import get_object_store
    from db.models import Frame, MachineVerdict, Object, OntologyClass, OntologyVersion, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    cls = next(c.id for c in onto.classes if c.name == "pedestrian")
    maker = get_sessionmaker()
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    img = np.random.default_rng(5).integers(30, 220, size=(240, 320, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)

    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        uri = store.put_bytes(f"frames/{sid}/cam_f/{fid}.jpg", buf.tobytes(), "image/jpeg")
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri,
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()

        quadrants = ([("correct", "accepted")] * tp + [("correct", "rejected")] * fp
                     + [("incorrect", "accepted")] * fn + [("incorrect", "rejected")] * tn)
        oids = []
        for machine_says, human_state in quadrants:
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[10.0, 10.0, 90.0, 120.0],
                          conf=0.6, source="fused", state=human_state, attrs={}, version=1,
                          provenance={"flywheel": {"cycle_id": batch_id}}))
            await db.flush()
            db.add(Review(object_id=oid, reviewer="rev", user_id=None, action="review",
                          before={}, after={}, time_spent_ms=1000, ts_ns=ts))
            db.add(MachineVerdict(object_id=oid, judge="vlm", provider="anthropic",
                                  model_version="test-judge-1", verdict=machine_says,
                                  confidence=0.9, detail={}, batch_id=batch_id, ts_ns=ts))
            oids.append(str(oid))
        await db.commit()
    return oids


@requires_infra
def test_machine_verdicts_do_not_land_in_the_human_review_plane():
    """The invariant three other things depend on.

    precision_batch excludes reviewed objects from sampling, corpus_precision reads states a human moved,
    and the annotator scorecards count review rows. A judge writing into `review` would corrupt all three
    invisibly, because the rows look identical.
    """
    from sqlalchemy import func, select

    from db.models import MachineVerdict, Review
    from db.session import get_sessionmaker

    batch = f"judge-test-{uuid.uuid4().hex[:8]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            before = (await db.execute(select(func.count(Review.review_id)))).scalar()
        await _seed_judgeable_batch(batch)
        async with get_sessionmaker()() as db:
            verdicts = (await db.execute(select(func.count(MachineVerdict.verdict_id))
                                         .where(MachineVerdict.batch_id == batch))).scalar()
            after = (await db.execute(select(func.count(Review.review_id)))).scalar()
        # 20 machine verdicts were written; the review rows that appeared are the 20 the fixture seeded as
        # human rulings, not the machine's opinions duplicated into that table.
        assert verdicts == 20
        assert after - before == 20

    run_async(_flow())


@requires_infra
def test_judge_agreement_measures_the_judge_against_the_humans():
    from db.session import get_sessionmaker
    from services.labelops.vlm_review import judge_agreement

    batch = f"judge-test-{uuid.uuid4().hex[:8]}"

    async def _flow():
        await _seed_judgeable_batch(batch, tp=9, fp=3, fn=1, tn=7)
        async with get_sessionmaker()() as db:
            a = await judge_agreement(db, model_version="test-judge-1", batch_id=batch)
        assert a["confusion"]["tp"] == 9 and a["confusion"]["fp"] == 3
        assert a["confusion"]["tn"] == 7 and a["confusion"]["fn"] == 1
        assert a["sensitivity"] == round(9 / 10, 4)      # of the labels humans accepted, the judge caught 9
        assert a["specificity"] == round(7 / 10, 4)      # of the labels humans rejected, the judge caught 7
        assert a["usable"] is True

    run_async(_flow())


@requires_infra
def test_precision_refuses_to_correct_when_the_judge_is_unmeasured():
    """The failure this module exists to prevent: quoting the judge's agreement rate as label precision.

    With no human rulings to compare against, `corrected` must be null and the caveat must say what the raw
    figure actually is, rather than the raw figure quietly standing in for a measurement.
    """
    import cv2

    from core.storage import get_object_store
    from db.models import Frame, MachineVerdict, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.vlm_review import judged_precision

    batch = f"judge-unmeasured-{uuid.uuid4().hex[:8]}"
    model_version = f"lonely-judge-{uuid.uuid4().hex[:6]}"

    async def _seed_unreviewed():
        store = get_object_store()
        store.ensure_bucket()
        onto = get_ontology()
        cls = next(c.id for c in onto.classes if c.name == "pedestrian")
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        img = np.random.default_rng(6).integers(30, 220, size=(240, 320, 3), dtype=np.uint8)
        _ok, buf = cv2.imencode(".jpg", img)
        async with get_sessionmaker()() as db:
            if await db.get(OntologyVersion, onto.version) is None:
                db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
                await db.flush()
                for c in onto.classes:
                    db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                         india=c.india, map_to={}))
                await db.flush()
            uri = store.put_bytes(f"frames/{sid}/cam_f/{fid}.jpg", buf.tobytes(), "image/jpeg")
            db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                             ontology_version=onto.version))
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri,
                         width=320, height=240, quality=0.9, scene={}))
            await db.flush()
            for i in range(10):
                oid = uuid.uuid4()
                db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[10.0, 10.0, 90.0, 120.0],
                              conf=0.6, source="fused", state="auto_accept", attrs={}, version=1,
                              provenance={"flywheel": {"cycle_id": batch}}))
                await db.flush()
                db.add(MachineVerdict(object_id=oid, judge="vlm", provider="anthropic",
                                      model_version=model_version,
                                      verdict="correct" if i < 8 else "incorrect",
                                      confidence=0.9, detail={}, batch_id=batch, ts_ns=ts))
            await db.commit()

    async def _flow():
        await _seed_unreviewed()
        async with get_sessionmaker()() as db:
            r = await judged_precision(db, batch, model_version=model_version)
        assert r["judged"] == 10
        assert r["raw"]["p"] == 0.8
        assert r["corrected"] is None, "an unmeasured judge cannot yield a corrected precision"
        assert "not the label precision" in r["caveat"]

    run_async(_flow())


@requires_infra
def test_precision_is_corrected_once_the_judge_has_been_measured():
    from db.session import get_sessionmaker
    from services.labelops.vlm_review import judged_precision

    batch = f"judge-test-{uuid.uuid4().hex[:8]}"

    async def _flow():
        await _seed_judgeable_batch(batch, tp=9, fp=3, fn=1, tn=7)
        async with get_sessionmaker()() as db:
            r = await judged_precision(db, batch, model_version="test-judge-1")
        # The judge called 12 of 20 correct, so a naive implementation reports precision 0.6.
        assert r["raw"]["p"] == 0.6
        # Corrected through the judge's own measured error (sensitivity 0.9, specificity 0.7), the labels
        # are actually 50% correct. Quoting 0.6 would have overstated quality by ten points.
        #
        # A dict rather than a float: the correction now carries the judge's own uncertainty through, because
        # a point estimate built from three uncertain quantities reads as more settled than it is. On the
        # real corpus that mattered, where a measured sensitivity of 0.76 and specificity of 0.80 put the
        # estimator's denominator anywhere between 0.30 and 0.74.
        assert r["corrected"]["p"] == pytest.approx(0.5, abs=1e-3), r["corrected"]
        assert r["corrected"]["clamped"] is False

    run_async(_flow())


def test_the_verdict_vocabulary_is_closed():
    """A fourth verdict would silently bypass the check constraint and the agreement arithmetic."""
    assert set(VERDICTS) == {"correct", "incorrect", "unsure"}


@requires_infra
def test_the_judge_is_measured_on_the_population_it_is_being_applied_to():
    """A judge's error rate is a property of the crops it judged, not a constant.

    This defaulted to measuring the judge across every batch it had ever seen while computing the rate on
    one batch, which silently assumes the batches are alike. They are not: a judge that separates
    pedestrians from poles cleanly can be much weaker on autorickshaw against e_auto, and mixing the two
    populations produced a correction that belonged to neither.
    """
    from db.session import get_sessionmaker
    from services.labelops.vlm_review import judged_precision

    strong = f"judge-strong-{uuid.uuid4().hex[:8]}"
    weak = f"judge-weak-{uuid.uuid4().hex[:8]}"
    model = f"scoped-judge-{uuid.uuid4().hex[:6]}"

    async def _flow():
        # Same judge, same raw rate on both batches, but it is far less reliable on the second.
        await _seed_judgeable_batch(strong, tp=9, fp=3, fn=1, tn=7)
        await _seed_judgeable_batch(weak, tp=6, fp=4, fn=4, tn=6)
        async with get_sessionmaker()() as db:
            from sqlalchemy import update

            from db.models import MachineVerdict
            await db.execute(update(MachineVerdict)
                             .where(MachineVerdict.batch_id.in_((strong, weak)))
                             .values(model_version=model))
            await db.commit()
        async with get_sessionmaker()() as db:
            on_own = await judged_precision(db, weak, model_version=model)
            borrowed = await judged_precision(db, weak, model_version=model,
                                              agreement_batch_id=strong)

        # The raw rate is identical either way: only the judge's measured reliability differs.
        assert on_own["raw"]["p"] == borrowed["raw"]["p"]
        assert on_own["judge_agreement"]["sensitivity"] != borrowed["judge_agreement"]["sensitivity"]
        assert on_own["corrected"] != borrowed["corrected"], (
            "borrowing a judge's reliability from a different population must change the answer, "
            "or the scoping parameter is doing nothing")

    run_async(_flow())


@requires_infra
def test_precision_reports_superclass_agreement_beside_the_strict_rate():
    """Without this the headline number is the most misleading one the system can produce.

    Re-judging the 300-crop batch with a stronger model took the strict agreement rate from 0.80 to 0.51,
    which reads as "half the corpus is mislabelled". Of the 140 rejections behind that, 128 proposed another
    four-wheeler (sedan to SUV) and 2 said the label named the wrong kind of thing entirely. Superclass
    agreement is 0.958.

    Both are true and they answer different questions, so both are returned: the strict rate is what a
    fine-grained consumer gets, and the superclass rate is what a safety gate cares about.
    """
    import uuid as _uuid

    from db.models import MachineVerdict
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelops.vlm_review import judged_precision

    onto = get_ontology()
    sedan = next(c for c in onto.classes if c.name == "sedan")
    suv = next(c for c in onto.classes if c.name == "suv")
    pole = next(c for c in onto.classes if c.name == "pole")
    batch = f"superclass-{_uuid.uuid4().hex[:8]}"
    model = f"judge-{_uuid.uuid4().hex[:6]}"

    async def _flow():
        oids = await _seed_batch_objects(batch, sedan.id, 4)
        async with get_sessionmaker()() as db:
            ts = 1
            # one confirmed, two refinements (sedan -> suv), one real error (sedan -> pole)
            for oid, verdict, proposed in (
                (oids[0], "correct", None),
                (oids[1], "incorrect", suv.id),
                (oids[2], "incorrect", suv.id),
                (oids[3], "incorrect", pole.id),
            ):
                db.add(MachineVerdict(object_id=_uuid.UUID(oid), judge="vlm", provider="ollama",
                                      model_version=model, verdict=verdict, proposed_class_id=proposed,
                                      confidence=0.9, detail={"given_class": "sedan"},
                                      batch_id=batch, ts_ns=ts))
            await db.commit()

        async with get_sessionmaker()() as db:
            r = await judged_precision(db, batch, model_version=model)

        assert r["judged"] == 4
        assert r["raw"]["p"] == 0.25, "strict: only one exact-class agreement"
        assert r["raw_superclass"]["p"] == 0.75, "superclass: the two SUV calls are the right kind of thing"
        assert r["rejections"]["refinement_within_superclass"] == 2
        assert r["rejections"]["cross_superclass"] == 1
        assert "safety gate cares about" in r["rejections"]["note"]

    run_async(_flow())


async def _seed_batch_objects(batch_id: str, class_id: int, n: int) -> list[str]:
    """n objects stamped into a flywheel batch, so judged_precision can find them."""
    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    out = []
    async with get_sessionmaker()() as db:
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
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/f.jpg",
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()
        for _ in range(n):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=class_id, bbox=[1.0, 1.0, 30.0, 30.0],
                          conf=0.6, source="fused", state="auto_accept", attrs={}, version=1,
                          provenance={"flywheel": {"cycle_id": batch_id}}))
            out.append(str(oid))
        await db.commit()
    return out
