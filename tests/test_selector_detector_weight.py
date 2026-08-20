"""The selector was ranking by which detector shouts loudest.

Error-candidate signal entered the active-learning score as a bare max() over the raw scores of every
detector that flagged an object. That assumes the detectors emit commensurable numbers, and they do not:
`confident_learning` reports a probability, `policy_violation` and `critic_flag` use hand-assigned
constants, and `near_dup_inconsistent` was reporting frame similarity, which cannot fall below its own 0.96
gate and therefore beat everything else on every object it touched.

Weighting by measured precision fixes the comparison and closes the loop: judging a detector in the error
queue changes what the selector does next.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.activelearn.selector import UNMEASURED_DETECTOR_WEIGHT

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


async def _seed(kind: str, confirmed: int, dismissed: int) -> None:
    """A detector with a known verdict record, so its weight is computable."""
    from core.timebase import now_ns, seconds_to_ns
    from db.models import ErrorCandidate, Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cls = next(c.id for c in onto.classes if c.name == "pedestrian")
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
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
        for i in range(confirmed + dismissed):
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fid, class_id=cls, bbox=[1.0, 1.0, 30.0, 30.0],
                          conf=0.6, source="fused", state="auto_accept", attrs={}, provenance={}, version=1))
            await db.flush()
            db.add(ErrorCandidate(object_id=oid, kind=kind, score=0.5, proposed_label=None, detail={},
                                  status="confirmed_error" if i < confirmed else "dismissed"))
        await db.commit()


@requires_infra
def test_an_unjudged_detector_is_treated_as_a_coin_not_as_reliable():
    """The old behaviour weighted every detector at 1.0, which is the claim that an unevaluated detector
    predicts perfectly. Zero would be defensible and useless: it would silence every detector in this
    corpus at once, since 298,529 candidates carry one verdict between them."""
    from db.session import get_sessionmaker
    from services.activelearn.selector import _detector_weights

    kind = f"unjudged_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed(kind, confirmed=0, dismissed=0)
        async with get_sessionmaker()() as db:
            w = await _detector_weights(db)
        assert kind not in w, "a detector with no verdicts has no measured weight"
        assert 0.0 < UNMEASURED_DETECTOR_WEIGHT < 1.0

    run_async(_flow())


@requires_infra
def test_a_detector_that_is_usually_wrong_is_weighted_below_one_that_is_usually_right():
    """The whole point: a detector earns its place in the queue by being confirmed."""
    from db.session import get_sessionmaker
    from services.activelearn.selector import _detector_weights

    good = f"good_{uuid.uuid4().hex[:6]}"
    bad = f"bad_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed(good, confirmed=45, dismissed=5)
        await _seed(bad, confirmed=5, dismissed=45)
        async with get_sessionmaker()() as db:
            w = await _detector_weights(db)
        assert w[good] > w[bad]
        assert w[good] > 0.7 and w[bad] < 0.3

    run_async(_flow())


@requires_infra
def test_a_small_sample_cannot_outrank_a_large_one_at_the_same_rate():
    """Why the weight is the Wilson lower bound and not the point estimate.

    Nine confirmations out of ten and nine hundred out of a thousand are both 0.9, but only the second has
    been demonstrated. Ranking on the point estimate would treat them as equals and let a detector buy its
    way up the queue with ten verdicts.
    """
    from db.session import get_sessionmaker
    from services.activelearn.selector import _detector_weights

    small = f"small_{uuid.uuid4().hex[:6]}"
    large = f"large_{uuid.uuid4().hex[:6]}"

    async def _flow():
        await _seed(small, confirmed=27, dismissed=3)     # 0.9 from 30
        await _seed(large, confirmed=270, dismissed=30)   # 0.9 from 300
        async with get_sessionmaker()() as db:
            w = await _detector_weights(db)
        assert w[large] > w[small], "the same rate from more evidence must weigh more"

    run_async(_flow())


@requires_infra
def test_the_weight_actually_reaches_the_selector_score():
    """A weight nothing consumes is decoration. This is the wiring assertion."""
    import inspect

    from services.activelearn import selector

    src = inspect.getsource(selector.score_candidates)
    assert "_detector_weights" in src, "the selector must consult the detector weights"
    assert "UNMEASURED_DETECTOR_WEIGHT" in src, "and fall back explicitly for unjudged detectors"
    # and the old shape, a bare max over raw scores, must be gone
    assert "max(err_scores.get(str(oid), 0.0), float(sc))" not in src
