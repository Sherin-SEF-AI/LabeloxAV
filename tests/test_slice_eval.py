"""Per-slice metrics are computed from the prediction plane, not typed by the caller.

The defect: services/verdyx/run.py::record_evaluation took `per_slice` as a caller-supplied dict, and nothing
in the codebase produced it. The protected-slice safety gate ("pedestrian_night", "autorickshaw_glare") was
therefore only as trustworthy as the JSON someone posted, with no causal link to the model being judged.

These tests seed a real gold set and a real inference run, then assert the slice numbers follow the data."""
from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    from core.config import get_settings
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


async def _seed(db, *, scene: dict, class_name: str, n_gt: int, n_hit: int):
    """One session/frame with `n_gt` gold boxes of `class_name`, of which `n_hit` are predicted correctly."""
    from db.models import Frame, GoldSet, InferenceRun, ModelRegistry, Object, Prediction
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name(class_name).id
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    db.add(DbSession(session_id=sid, vehicle_id="SLICE-01", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/y.jpg",
                 width=1280, height=960, quality=1.0, scene=scene))
    await db.flush()

    gold_ids = []
    for i in range(n_gt):
        oid = uuid.uuid4()
        x = 10.0 + i * 120.0
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[x, 10.0, x + 100.0, 210.0],
                      conf=1.0, attrs={}, source="human", state="accepted"))
        gold_ids.append(str(oid))

    mv = f"slice-test-{uuid.uuid4().hex[:8]}"
    db.add(ModelRegistry(model_version=mv, task="detection", gold_metrics={}, is_champion=False))
    await db.flush()
    run = InferenceRun(model_version=mv, gold_id=None, frame_count=1, params={}, status="done")
    db.add(run)
    await db.flush()

    # n_hit predictions land exactly on the first n_hit gold boxes; the rest are simply absent (misses).
    for i in range(n_hit):
        x = 10.0 + i * 120.0
        db.add(Prediction(run_id=run.run_id, frame_id=fid, class_id=cid,
                          bbox=[x, 10.0, x + 100.0, 210.0], conf=0.9))

    gold = GoldSet(gold_id=f"gold-{uuid.uuid4().hex[:12]}", name="slice-test",
                   object_ids=gold_ids, n_objects=len(gold_ids), n_frames=1,
                   ontology_version=onto.version, spec={})
    db.add(gold)
    await db.commit()
    return gold.gold_id, str(run.run_id)


@requires_infra
async def test_recall_follows_the_data_for_a_scene_and_class_slice():
    from db.session import get_sessionmaker
    from services.verdyx.slice_eval import compute_slice_metrics

    async with get_sessionmaker()() as db:
        # four night pedestrians, three detected
        gold_id, run_id = await _seed(db, scene={"time_of_day": "night"}, class_name="pedestrian",
                                      n_gt=4, n_hit=3)
        out = await compute_slice_metrics(db, gold_id, run_id=run_id, slice_ids=["pedestrian_night"])

    m = out["pedestrian_night"]
    assert m["measured"] is True
    assert m["support"] == 4 and m["tp"] == 3 and m["fn"] == 1
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)


@requires_infra
async def test_a_daytime_frame_does_not_enter_the_night_slice():
    from db.session import get_sessionmaker
    from services.verdyx.slice_eval import compute_slice_metrics

    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, scene={"time_of_day": "day"}, class_name="pedestrian",
                                      n_gt=3, n_hit=3)
        out = await compute_slice_metrics(db, gold_id, run_id=run_id, slice_ids=["pedestrian_night"])

    # the scene predicate excludes every frame, so there is no evidence rather than a perfect score
    assert out["pedestrian_night"]["measured"] is False
    assert out["pedestrian_night"]["support"] == 0


@requires_infra
async def test_unevidenced_slice_is_unmeasured_not_a_silent_pass():
    # The gate must be able to tell "no data" from "measured and fine": reporting 1.0 here would let a
    # protected slice with no gold coverage wave a challenger through.
    from db.session import get_sessionmaker
    from services.verdyx.slice_eval import compute_slice_metrics

    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, scene={"time_of_day": "night"}, class_name="pedestrian",
                                      n_gt=2, n_hit=2)
        out = await compute_slice_metrics(db, gold_id, run_id=run_id,
                                          slice_ids=["autorickshaw_glare"])

    assert out["autorickshaw_glare"]["measured"] is False


@requires_infra
async def test_slice_carries_the_map_key_the_gate_compares_on():
    from db.session import get_sessionmaker
    from services.verdyx.slice_eval import compute_slice_metrics

    async with get_sessionmaker()() as db:
        gold_id, run_id = await _seed(db, scene={"time_of_day": "night"}, class_name="pedestrian",
                                      n_gt=4, n_hit=4)
        out = await compute_slice_metrics(db, gold_id, run_id=run_id, slice_ids=["pedestrian_night"])

    m = out["pedestrian_night"]
    # services/verdyx/verdict.py::slice_verdict reads "map"; without it every slice looks unmeasured.
    assert "map" in m and m["map"] == m["ap50"]
    assert m["map"] > 0.9   # every gold box detected at high confidence


@requires_infra
async def test_unknown_run_or_gold_is_an_error_not_an_empty_pass():
    from db.session import get_sessionmaker
    from services.verdyx.slice_eval import compute_slice_metrics

    async with get_sessionmaker()() as db:
        out = await compute_slice_metrics(db, "gold-does-not-exist", run_id=str(uuid.uuid4()))
    assert "error" in out
