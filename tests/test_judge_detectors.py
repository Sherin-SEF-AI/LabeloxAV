"""Measuring a detector with the judge instead of a person.

298,528 candidates carry one human verdict between them, so every detector reports as unmeasured and the
selector weights all of them at a placeholder. The judge answers the same question in a different shape:
a detector's flag claims the label is suspect, and the judge says whether the label is correct.

The tests are shaped around the four ways this goes quietly wrong: sampling the detector's best work,
letting a machine verdict into the human plane, counting fine-class refinements as real finds, and letting a
machine estimate outrank a human ruling.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings


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


async def _seed_candidates(kind: str, n: int, *, class_name: str = "sedan",
                           real_image: bool = False) -> list[str]:
    """n pending candidates of one detector kind, on real objects.

    `real_image` stores an actual encoded frame, which the judging path needs: without readable pixels every
    candidate is skipped as unreadable and the test would pass while measuring nothing.
    """
    from core.timebase import now_ns, seconds_to_ns
    from db.models import ErrorCandidate, Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = next(c.id for c in onto.classes if c.name == class_name)
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
        img_uri = "s3://x/f.jpg"
        if real_image:
            import cv2
            import numpy as np

            from core.storage import get_object_store

            store = get_object_store()
            store.ensure_bucket()
            arr = np.random.default_rng(3).integers(40, 200, size=(240, 320, 3), dtype=np.uint8)
            _ok, buf = cv2.imencode(".jpg", arr)
            img_uri = store.put_bytes(f"frames/{sid}/cam_f/{fid}.jpg", buf.tobytes(), "image/jpeg")
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=img_uri,
                     width=320, height=240, quality=0.9, scene={}))
        await db.flush()
        for i in range(n):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[1.0, 1.0, 30.0, 30.0],
                          conf=0.6, source="fused", state="auto_accept", attrs={}, provenance={}, version=1))
            await db.flush()
            db.add(ErrorCandidate(object_id=oid, kind=kind, score=i / max(1, n),
                                  proposed_label=None, detail={}, status="pending"))
            out.append(str(oid))
        await db.commit()
    return out


# --- sampling ------------------------------------------------------------------------------------


@requires_infra
def test_the_sample_is_random_not_the_detector_s_best_work():
    """Judging a detector's highest-scored flags measures how good it is when most confident, which is not
    what the queue needs to know. Same reason precision_batch samples randomly."""
    import inspect

    from services.errordetect import judge_detectors

    src = inspect.getsource(judge_detectors.sample_candidates)
    assert "func.random()" in src
    assert "score.desc()" not in src, "ranking by score would measure the detector's best case"


@requires_infra
def test_sampling_only_reaches_undecided_candidates_of_that_detector():
    from db.session import get_sessionmaker
    from services.errordetect.judge_detectors import sample_candidates

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"
    other = f"test_other_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed_candidates(kind, 6)
        await _seed_candidates(other, 6)
        async with get_sessionmaker()() as db:
            got = await sample_candidates(db, kind, 50)
        assert len(got) == 6
        assert all(g["class_name"] == "sedan" for g in got)

    run_async(_flow())


# --- the plane separation ---------------------------------------------------------------------------


@requires_infra
def test_a_machine_verdict_never_touches_the_queue_s_own_status():
    """A machine confirming its own detector into error_candidate.status would corrupt the one honest
    signal in that table, and the human-verdict precision would then be measuring the machine."""
    import inspect

    from services.errordetect import judge_detectors

    import re

    src = inspect.getsource(judge_detectors)

    # Reading `status == "pending"` to find undecided candidates is fine and necessary; the regex requires a
    # single `=` so the comparison does not match. What must never appear is a write.
    assert not re.search(r"\.status\s*=(?!=)", src), (
        "this module must not assign the queue's verdict column")
    for forbidden in ('values(status', 'status="confirmed_error"', "status='confirmed_error'",
                      'status="dismissed"', "status='dismissed'", "update(ErrorCandidate"):
        assert forbidden not in src, f"this module must not write the queue's verdict column ({forbidden})"
    assert "Review(" not in src, "and it must never write into the human review table"


@requires_infra
def test_judged_verdicts_are_stamped_as_a_detector_estimate():
    """They have to be identifiable as an estimate rather than mixed into the measurement batches."""
    from services.errordetect.judge_detectors import BATCH_PREFIX

    assert BATCH_PREFIX and BATCH_PREFIX != "judge-calibration"
    assert not BATCH_PREFIX.startswith("precision-")


# --- what counts as a find ---------------------------------------------------------------------------


class _StubJudge:
    """A judge whose answer is scripted, so the arithmetic can be tested without a model.

    Exposes chat_json because that is what `_ask` duck-types on, the same shape as the real cloud and local
    clients.
    """

    model = "stub-judge"

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def chat_json(self, prompt, *, model=None, image_jpeg=None, temperature=0.0):
        r = self._replies[self.calls % len(self._replies)]
        self.calls += 1
        return r


@requires_infra
def test_a_fine_class_refinement_is_not_counted_as_finding_an_error(tmp_path):
    """A detector flagging a sedan that is really an SUV is technically right and useless to a safety gate.

    This corpus runs 128 refinements to 2 cross-superclass errors, so counting every confirmation the same
    way would rate every detector far above its usefulness to the queue.
    """
    from db.session import get_sessionmaker
    from services.errordetect.judge_detectors import judge_detector

    kind = f"test_kind_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed_candidates(kind, 4, real_image=True)
        # Two refinements (sedan -> suv), one real error (sedan -> pole), one confirmation of the label.
        judge = _StubJudge([
            {"verdict": "incorrect", "correct_class": "suv", "confidence": 0.9, "reason": "roof rails"},
            {"verdict": "incorrect", "correct_class": "suv", "confidence": 0.9, "reason": "ground clearance"},
            {"verdict": "incorrect", "correct_class": "pole", "confidence": 0.9, "reason": "not a vehicle"},
            {"verdict": "correct", "correct_class": None, "confidence": 0.9, "reason": "a sedan"},
        ])
        async with get_sessionmaker()() as db:
            r = await judge_detector(db, kind, n=4, client=judge, model_version="stub-judge")

        assert r["judged"] == 4
        assert r["confirmed"] == 3 and r["dismissed"] == 1
        assert r["refinements_within_superclass"] == 2
        assert r["cross_superclass"] == 1
        # Strict says the detector was right 3 times in 4; the useful reading says once.
        assert r["precision_strict"]["p"] == 0.75
        assert r["precision_cross_superclass"]["p"] == 0.25
        assert r["estimate"] is True

    run_async(_flow())


@requires_infra
def test_precision_is_reported_both_strictly_and_cross_superclass():
    import inspect

    from services.errordetect import judge_detectors

    src = inspect.getsource(judge_detectors.judge_detector)
    assert '"precision_strict"' in src and '"precision_cross_superclass"' in src
    assert '"refinements_within_superclass"' in src


# --- the weighting tiers ------------------------------------------------------------------------------


@requires_infra
def test_a_human_verdict_outranks_a_machine_estimate_for_the_same_detector():
    """The tiers are not equally trustworthy. A model grading a model is weaker evidence than somebody
    looking, and if the machine estimate could overwrite a human ruling the ranking would get worse the more
    it was judged."""
    import inspect

    from services.activelearn import selector

    src = inspect.getsource(selector._detector_weights)
    idx_machine = src.index("machine_detector_weights(db)")
    idx_human = src.index("detector_precision(db)")
    assert idx_machine < idx_human, (
        "the machine tier must be written first so the human tier overwrites it")


@requires_infra
def test_an_unjudged_detector_still_falls_back_to_the_unproven_default():
    from services.activelearn.selector import UNMEASURED_DETECTOR_WEIGHT

    assert 0.0 < UNMEASURED_DETECTOR_WEIGHT < 1.0


@requires_infra
def test_a_machine_weight_needs_enough_judged_candidates_to_count():
    """Otherwise a detector judged three times would enter the ranking on an interval spanning everything."""
    from db.session import get_sessionmaker
    from services.errordetect.judge_detectors import MIN_JUDGED, machine_detector_weights

    assert MIN_JUDGED >= 20

    async def _flow():
        async with get_sessionmaker()() as db:
            weights = await machine_detector_weights(db)
        assert isinstance(weights, dict)
        assert all(0.0 <= v <= 1.0 for v in weights.values())

    run_async(_flow())


@requires_infra
def test_judging_an_object_in_a_second_batch_does_not_steal_it_from_the_first():
    """Found in production, silently.

    Uniqueness was (object_id, judge, model_version), so when the detector sample hit objects already in the
    calibration set the upsert rewrote their batch_id and the calibration lost nine of its 247 verdicts.
    Nothing errored; it would have carried on reporting a sensitivity over a set that had quietly shrunk.
    Every reader here filters by batch, so the batch has to be part of the key.
    """
    import uuid as _uuid

    from core.timebase import now_ns
    from db.models import MachineVerdict
    from db.session import get_sessionmaker
    from sqlalchemy import func, select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    async def _flow():
        oids = await _seed_candidates(f"t_{_uuid.uuid4().hex[:6]}", 1)
        oid = _uuid.UUID(oids[0])
        model = f"m-{_uuid.uuid4().hex[:6]}"

        async def _write(batch: str, verdict: str):
            async with get_sessionmaker()() as db:
                await db.execute(pg_insert(MachineVerdict).values(
                    object_id=oid, judge="vlm", provider="ollama", model_version=model,
                    verdict=verdict, confidence=0.9, detail={"batch": batch},
                    batch_id=batch, ts_ns=now_ns(),
                ).on_conflict_do_update(
                    constraint="uq_machine_verdict_object_judge_batch",
                    set_={"verdict": verdict, "detail": {"batch": batch}, "ts_ns": now_ns()}))
                await db.commit()

        await _write("batch-one", "correct")
        await _write("batch-two", "incorrect")

        async with get_sessionmaker()() as db:
            rows = (await db.execute(
                select(MachineVerdict.batch_id, MachineVerdict.verdict)
                .where(MachineVerdict.object_id == oid, MachineVerdict.model_version == model))).all()
        by_batch = dict(rows)
        assert by_batch == {"batch-one": "correct", "batch-two": "incorrect"}, (
            "each batch must keep its own verdict for the same object")

        # And re-running the same batch still updates in place rather than duplicating.
        await _write("batch-one", "unsure")
        async with get_sessionmaker()() as db:
            n = (await db.execute(select(func.count(MachineVerdict.verdict_id)).where(
                MachineVerdict.object_id == oid, MachineVerdict.model_version == model))).scalar()
        assert n == 2, "re-running a batch must not add a row"

    run_async(_flow())


@requires_infra
def test_skipping_already_judged_objects_is_scoped_to_the_batch():
    """The mirror of the same mistake. Without the batch filter an object judged elsewhere by the same model
    reads as already done, and the batch ends up with a hole nothing reports."""
    import inspect

    from services.labelops import vlm_review

    src = inspect.getsource(vlm_review.prereview_batch)
    assert "MachineVerdict.batch_id == batch_id" in src
