"""Explore workspace: SQL predicate, faceted counts, bulk tagging, embeddings projection, saved views.

Requires infra (DB). Everything is scoped to one throwaway session seeded here and torn down at the end, so a
polluted corpus cannot make these assertions flap (the isolation lesson from the earlier suite).
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.timebase import now_ns, seconds_to_ns


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed():
    """One session, one frame, three objects with distinct classes/states/confidences."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    # pick three real ontology class ids so the class facet can name them
    cids = [c.id for c in list(onto.classes)[:3]] if hasattr(onto, "classes") else [1, 2, 3]
    while len(cids) < 3:
        cids.append(cids[-1])

    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="EXPLORE-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="TESTCITY", sensors={},
                         ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f",
                     img_uri=f"s3://x/{sid}.jpg", width=640, height=480, quality=0.9,
                     scene={"weather": "rain", "time_of_day": "night"}))
        await db.flush()  # frame before its objects (raw FK columns, no ORM relationship ordering)
        oids = []
        for i, (cid, state, conf) in enumerate(
                [(cids[0], "accepted", 0.95), (cids[1], "review", 0.42), (cids[2], "accepted", 0.75)]):
            oid = uuid.uuid4()
            oids.append(oid)
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[i * 10.0, 0.0, i * 10 + 5.0, 5.0],
                          conf=conf, source="fused", state=state, attrs={}, provenance={}))
        await db.commit()
    return str(sid), str(fid), [str(o) for o in oids], cids


async def _teardown(sid: str):
    from sqlalchemy import delete

    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        # frames/objects cascade from session
        await db.execute(delete(DbSession).where(DbSession.session_id == uuid.UUID(sid)))
        await db.commit()


@requires_infra
def test_predicate_facets_and_tags():
    from db.session import get_sessionmaker
    from services.explore.facets import object_facets
    from services.explore.query import frame_select, object_select
    from services.explore.tags import apply_tags, normalize_tags, tag_vocabulary

    async def run():
        sid, fid, oids, cids = await _seed()
        try:
            from sqlalchemy import func

            async with get_sessionmaker()() as db:
                base = {"session_id": sid}

                # --- predicate translates to SQL and scopes correctly
                n = (await db.execute(object_select(base, func.count()))).scalar_one()
                assert n == 3, f"expected the 3 seeded objects, got {n}"

                # state clause
                n_rev = (await db.execute(
                    object_select({**base, "states": ["review"]}, func.count()))).scalar_one()
                assert n_rev == 1

                # confidence clause
                n_hi = (await db.execute(
                    object_select({**base, "min_conf": 0.7}, func.count()))).scalar_one()
                assert n_hi == 2

                # scene axis clause (JSONB)
                n_rain = (await db.execute(
                    object_select({**base, "weather": ["rain"]}, func.count()))).scalar_one()
                assert n_rain == 3
                n_clear = (await db.execute(
                    object_select({**base, "weather": ["clear"]}, func.count()))).scalar_one()
                assert n_clear == 0

                # city clause forces the session join
                n_city = (await db.execute(
                    object_select({**base, "cities": ["TESTCITY"]}, func.count()))).scalar_one()
                assert n_city == 3

                # frame-level select with an object-level clause becomes an EXISTS
                n_f = (await db.execute(
                    frame_select({**base, "states": ["review"]}, func.count()))).scalar_one()
                assert n_f == 1, "the one frame has an object in review, so it matches once"

                # --- facets: each facet drops its own clause
                f = await object_facets(db, {**base, "states": ["review"]})
                assert f["total"] == 1, "total honours the full predicate"
                state_counts = {r["value"]: r["count"] for r in f["states"]}
                assert state_counts.get("accepted") == 2 and state_counts.get("review") == 1, (
                    "the state facet drops its own clause so every state stays selectable")
                assert sum(b["count"] for b in f["conf"]) == 1, "conf histogram honours the state clause"

                # --- tags: add, idempotency, any-of filtering, removal
                assert normalize_tags([" Night ", "NIGHT", "", "x" * 100]) == ["night", "x" * 64]

                r = await apply_tags(db, level="object", pred={**base, "states": ["review"]},
                                     add=["needs_relabel", "Night"])
                assert r["matched"] == 1 and r["added"] == ["needs_relabel", "night"]

                n_tag = (await db.execute(
                    object_select({**base, "tags": ["needs_relabel"]}, func.count()))).scalar_one()
                assert n_tag == 1

                # idempotent: re-applying does not duplicate
                await apply_tags(db, level="object", pred={**base, "tags": ["needs_relabel"]},
                                 add=["needs_relabel"])
                from db.models import Object
                tags = (await db.execute(
                    object_select({**base, "tags": ["needs_relabel"]}, Object.tags))).scalars().all()
                assert tags and sorted(tags[0]) == ["needs_relabel", "night"], f"got {tags}"

                # tag vocabulary sees it
                vocab = {t["tag"]: t["count"] for t in await tag_vocabulary(db, "object")}
                assert vocab.get("needs_relabel", 0) >= 1

                # removal
                await apply_tags(db, level="object", pred={**base, "tags": ["needs_relabel"]},
                                 remove=["night"])
                tags2 = (await db.execute(
                    object_select({**base, "tags": ["needs_relabel"]}, Object.tags))).scalars().all()
                assert tags2 and tags2[0] == ["needs_relabel"]

                # frame-level tagging works through the same path
                rf = await apply_tags(db, level="frame", pred=base, add=["golden"])
                assert rf["matched"] == 1
                n_ft = (await db.execute(
                    object_select({**base, "frame_tags": ["golden"]}, func.count()))).scalar_one()
                assert n_ft == 3, "all 3 objects sit on the one golden frame"
        finally:
            await _teardown(sid)

    run_async(run())


@requires_infra
def test_projection_fit_persists_and_reads_back():
    from db.models import ObjectEmbedding
    from db.session import get_sessionmaker
    from services.curation.projection import (
        delete_projection,
        fit_projection,
        projection_points,
    )

    async def run():
        sid, fid, oids, _ = await _seed()
        pid = None
        try:
            rng = np.random.default_rng(7)
            async with get_sessionmaker()() as db:
                for oid in oids:
                    db.add(ObjectEmbedding(object_id=uuid.UUID(oid),
                                           dino_vec=rng.normal(size=768).astype(float).tolist(),
                                           model_versions={"test": "1"}))
                await db.commit()

            async with get_sessionmaker()() as db:
                # 3 points is below UMAP's neighbourhood floor, so this must fall back to PCA and say so
                res = await fit_projection(db, kind="object", space="dino", session_id=sid)
                assert res["n"] == 3, res
                assert res["method"] == "pca", "a 3-point fit degrades to PCA and reports it honestly"
                pid = res["projection_id"]

                pts = await projection_points(db, pid)
                assert len(pts["points"]) == 3
                p0 = pts["points"][0]
                assert {"id", "x", "y", "cluster"} <= set(p0), p0
                assert "class_id" in p0 and "state" in p0, "points carry the attributes the map colours by"
        finally:
            if pid:
                async with get_sessionmaker()() as db:
                    await delete_projection(db, pid)
            await _teardown(sid)

    run_async(run())


@requires_infra
def test_object_siglip_is_rejected():
    """Object crops are embedded with DINOv3 only; asking for siglip must fail loudly, not silently return
    an empty or wrong-space map."""
    from db.session import get_sessionmaker
    from services.curation.projection import fit_projection

    async def run():
        async with get_sessionmaker()() as db:
            with pytest.raises(ValueError):
                await fit_projection(db, kind="object", space="siglip")

    run_async(run())


@requires_infra
def test_eval_patches_capture_confusion_and_misses():
    """A cross-class confusion must land in an off-diagonal cell (both class ids set), an unmatched
    prediction must be an fp with no gt class, and an unmatched gold object must be an fn."""
    import uuid as _uuid

    from db.models import GoldSet, InferenceRun, ModelRegistry, Object, Prediction
    from db.session import get_sessionmaker
    from services.analytics.evaluation import (
        cell_patches,
        confusion_cells,
        delete_evaluation,
        evaluate_gold_patches,
    )

    async def run():
        sid, fid, oids, cids = await _seed()
        eval_id = None
        gold_id = f"gold-test-{_uuid.uuid4().hex[:8]}"
        try:
            from sqlalchemy import update

            async with get_sessionmaker()() as db:
                # Make the 3 seeded objects the GOLD (human truth) on this frame.
                await db.execute(update(Object).where(Object.object_id.in_([_uuid.UUID(o) for o in oids]))
                                 .values(source="human", state="accepted"))
                db.add(GoldSet(gold_id=gold_id, name="explore-eval-test", spec={},
                               object_ids=[str(o) for o in oids], n_objects=3, n_frames=1,
                               ontology_version="test"))

                # Predictions on the same frame, in the immutable prediction plane under one inference run:
                #  - exact overlap of gold[0] but WRONG class  -> off-diagonal confusion
                #  - exact overlap of gold[1] with RIGHT class -> tp
                #  - a box overlapping nothing                 -> fp
                #  gold[2] gets no prediction                  -> fn
                model_v = f"m-explore-{_uuid.uuid4().hex[:8]}"
                db.add(ModelRegistry(model_version=model_v, weights_uri="s3://w.pt"))
                await db.flush()
                run = InferenceRun(run_id=_uuid.uuid4(), model_version=model_v, gold_id=gold_id,
                                   status="complete", params={"imgsz": 640})
                db.add(run)
                await db.flush()
                run_id = str(run.run_id)
                db.add(Prediction(run_id=run.run_id, frame_id=_uuid.UUID(fid), class_id=cids[2],
                                  bbox=[0.0, 0.0, 5.0, 5.0], conf=0.9))
                db.add(Prediction(run_id=run.run_id, frame_id=_uuid.UUID(fid), class_id=cids[1],
                                  bbox=[10.0, 0.0, 15.0, 5.0], conf=0.8))
                db.add(Prediction(run_id=run.run_id, frame_id=_uuid.UUID(fid), class_id=cids[0],
                                  bbox=[500.0, 400.0, 520.0, 420.0], conf=0.7))
                await db.commit()

            async with get_sessionmaker()() as db:
                res = await evaluate_gold_patches(db, gold_id, run_id=run_id, iou_thr=0.5)
                eval_id = res["eval_id"]
                assert res["tp"] == 1, res
                assert res["fn"] == 1, res
                # one wrong-class match + one box matching nothing
                assert res["fp"] == 2, res

                cells = (await confusion_cells(db, eval_id))["cells"]
                off_diag = [c for c in cells
                            if c["gt_class_id"] is not None and c["pred_class_id"] is not None
                            and c["gt_class_id"] != c["pred_class_id"]]
                assert off_diag, f"expected an off-diagonal confusion cell, got {cells}"

                # drilling into that cell returns the real object with a crop url
                cell = off_diag[0]
                got = await cell_patches(db, eval_id, gt_class_id=cell["gt_class_id"],
                                         pred_class_id=cell["pred_class_id"])
                assert got["count"] >= 1
                # the confusion patch is now a prediction, cropped from the prediction plane
                assert got["patches"][0]["crop_url"].startswith("/api/predictions/")

                # misses are recorded with no predicted class
                misses = await cell_patches(db, eval_id, outcome="fn")
                assert misses["count"] == 1
                assert misses["patches"][0]["pred_class_id"] is None
        finally:
            async with get_sessionmaker()() as db:
                if eval_id:
                    await delete_evaluation(db, eval_id)
                g = await db.get(GoldSet, gold_id)
                if g:
                    await db.delete(g)
                    await db.commit()
            await _teardown(sid)

    run_async(run())


def test_export_spec_carries_tag_and_scene_clauses():
    """A view saved purely on tags must not degrade to an unfiltered export. slice_to_export_spec predates
    tags, and silently dropping the clause would export the whole corpus while claiming to be the cohort."""
    from types import SimpleNamespace

    from services.curation.slices import slice_to_export_spec

    row = SimpleNamespace(name="night-rain-audit",
                          predicate={"tags": ["night_rain_audit"], "weather": ["rain"],
                                     "states": ["accepted"], "sources": ["fused"], "max_conf": 0.9})
    out = slice_to_export_spec(row)
    assert out["tag_filter"] == {"tags": ["night_rain_audit"]}, out
    assert out["scene_filter"] == {"weather": ["rain"]}, out
    assert out["spec"]["states"] == ["accepted"]
    assert out["sources"] == ["fused"] and out["max_conf"] == 0.9
    # everything in this predicate is expressible, so nothing is silently dropped
    assert out["unsupported"] == [], out

    # an unknown clause must be reported, not ignored
    row2 = SimpleNamespace(name="x", predicate={"some_future_clause": ["v"]})
    assert slice_to_export_spec(row2)["unsupported"] == ["some_future_clause"]
