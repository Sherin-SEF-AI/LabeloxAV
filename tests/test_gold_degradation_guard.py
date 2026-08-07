"""A gold set that has lost its objects must stop producing numbers.

`GoldSet.object_ids` is a sealed JSONB membership list, not a set of foreign keys. Sealing protects the list
from being edited; it does not protect the rows it names. A fixture purge deleted objects, nothing cascaded,
nothing warned, and the largest sealed set in this corpus held 47 of its 400 for months while every
evaluation against it reported confident figures that looked exactly like figures about 400 objects.

The failure has no symptom by construction: fewer ground-truth boxes is a smaller evaluation, not an error.
So the guard has to be a refusal at the point of measurement rather than a warning somewhere a reader might
look.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns
from db.models import Frame, GoldSet, ModelRegistry, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.govern.gold_eval import MIN_RESOLVABLE_FRACTION, _gold_population, evaluate_on_gold

pytestmark = pytest.mark.db


async def _seal(db, *, n_real: int, n_phantom: int, model: str | None = None) -> str:
    """A sealed set naming `n_real` objects that exist and `n_phantom` that never will.

    Phantom ids stand in for objects deleted after sealing. Nothing distinguishes the two cases from the
    set's own point of view, which is the whole problem.
    """
    onto = get_ontology()
    cid = onto.by_name("truck").id
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-GOLDGUARD", start_ts_ns=0, end_ts_ns=1,
                     ontology_version=onto.version)
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/test.jpg", width=100, height=100)
    db.add(frame)

    ids: list[str] = []
    for _ in range(n_real):
        o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=cid,
                   bbox=[10.0, 10.0, 50.0, 50.0], conf=1.0, source="human", state="accepted")
        db.add(o)
        await db.flush()
        ids.append(str(o.object_id))
    ids += [str(uuid.uuid4()) for _ in range(n_phantom)]

    gold_id = f"gold-guard-{uuid.uuid4().hex[:12]}"
    db.add(GoldSet(gold_id=gold_id, name="guard", object_ids=ids, n_objects=len(ids), n_frames=1,
                   ontology_version=onto.version))
    if model and not await db.get(ModelRegistry, model):
        db.add(ModelRegistry(model_version=model, weights_uri="s3://nonexistent-weights.pt"))
    await db.commit()
    return gold_id


async def test_population_counts_what_exists_not_what_was_sealed():
    async with get_sessionmaker()() as db:
        gold_id = await _seal(db, n_real=3, n_phantom=7)
        declared, resolvable = await _gold_population(db, gold_id)
        assert (declared, resolvable) == (10, 3)


async def test_a_degraded_set_is_refused_before_any_gpu_work():
    """The registered model points at weights that do not exist, so reaching the val pass at all would raise
    rather than return this error. Getting the refusal proves it short-circuited first."""
    model = "m-goldguard-degraded"
    async with get_sessionmaker()() as db:
        gold_id = await _seal(db, n_real=1, n_phantom=9, model=model)
        out = await evaluate_on_gold(db, model, gold_id)

        assert out is not None and "error" in out
        assert out["gold_declared"] == 10
        assert out["gold_resolvable"] == 1
        assert out.get("map50") is None, "a refusal must not carry a score"
        assert "47" not in str(out.get("detail", "")), "the message must describe this set, not the example"


async def test_a_healthy_set_is_not_blocked_by_the_guard():
    """The floor is not 100%: one object removed by a legitimate correction must not halt the gate. This set
    passes the guard and then fails later on its fake weights, which is the guard letting it through."""
    model = "m-goldguard-healthy"
    async with get_sessionmaker()() as db:
        gold_id = await _seal(db, n_real=10, n_phantom=0, model=model)
        declared, resolvable = await _gold_population(db, gold_id)
        assert resolvable / declared >= MIN_RESOLVABLE_FRACTION

        out = await evaluate_on_gold(db, model, gold_id)
        assert out is None or out.get("error") != "gold set is too degraded to score against"


async def test_the_floor_sits_well_above_the_degradation_that_went_unnoticed():
    """The set that shipped numbers for months resolved 12%. A floor at or below that would have permitted
    every one of them."""
    assert MIN_RESOLVABLE_FRACTION > 0.12
    assert MIN_RESOLVABLE_FRACTION < 1.0


async def test_a_missing_gold_set_reports_an_empty_population_rather_than_raising():
    async with get_sessionmaker()() as db:
        assert await _gold_population(db, "gold-does-not-exist") == (0, 0)
