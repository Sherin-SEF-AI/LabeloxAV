"""M4.0 active learning: within a human-hour budget the selector returns a ranked batch that is uncertain,
diverse, and rich in rare cases rather than redundant easy frames; and the loop counts new signal and
fires a burst fine-tune. Constructed pool: easy (high-conf, agreeing, common, near-duplicate embeddings)
vs uncertain (mid-conf, disagreeing) vs rare (fallback/india class). Single asyncio.run."""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
def test_al_selects_uncertain_diverse_rare_over_easy():
    from db.models import Frame, Object, ObjectEmbedding
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.activelearn.budget import select_batch

    sid = uuid.uuid4()
    rng = np.random.default_rng(11)

    def vec(seed):
        v = np.random.default_rng(seed).standard_normal(768).astype("float32")
        return (v / np.linalg.norm(v)).tolist()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="AL-TEST", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=640, height=480)
            db.add(f)
            await db.flush()

            easy, unc, rare = [], [], []
            # 4 easy: high conf, agreement, common class, all near-duplicate embeddings (one diverse signal)
            for i in range(4):
                o = Object(frame_id=f.frame_id, class_id=1, bbox=[i, 0, i + 5, 5], conf=0.985,
                           state="auto_accept", source="auto_accept", provenance={"agreement": True})
                db.add(o); await db.flush(); easy.append(o.object_id)
                db.add(ObjectEmbedding(object_id=o.object_id, dino_vec=vec(100), model_versions={}))  # identical
            # 3 uncertain: mid-conf in the band, path disagreement, common class, diverse embeddings
            for i in range(3):
                o = Object(frame_id=f.frame_id, class_id=2, bbox=[i, 10, i + 5, 15], conf=0.73,
                           state="review", source="fused", provenance={"agreement": False, "mask_box_disagree": True})
                db.add(o); await db.flush(); unc.append(o.object_id)
                db.add(ObjectEmbedding(object_id=o.object_id, dino_vec=vec(200 + i), model_versions={}))
            # 2 rare: fallback/india class, diverse embeddings
            for i in range(2):
                o = Object(frame_id=f.frame_id, class_id=5, bbox=[i, 20, i + 5, 25], conf=0.7,
                           state="review", source="fused", provenance={"agreement": True})
                db.add(o); await db.flush(); rare.append(o.object_id)
                db.add(ObjectEmbedding(object_id=o.object_id, dino_vec=vec(300 + i), model_versions={}))
            await db.commit()

            res = await select_batch(db, budget_hours=0.05, session_id=str(sid), dedup_cos=0.9)  # ~6 items at 30s each
            items = res["items"]
            assert items, "selector returned nothing"
            top_ids = [it["object_id"] for it in items]

            # the uncertain + rare items outrank the easy ones
            ranks = {oid: i for i, oid in enumerate(top_ids)}
            best_easy = min((ranks.get(str(o), 999) for o in easy), default=999)
            worst_interesting = max((ranks.get(str(o), -1) for o in unc + rare if str(o) in ranks), default=-1)
            assert worst_interesting < best_easy, "an easy item outranked an uncertain/rare item"

            # near-duplicate easy embeddings are suppressed: not all 4 identical-embedding easies are picked
            picked_easy = sum(1 for o in easy if str(o) in ranks)
            assert picked_easy <= 2, f"duplicate easies not suppressed ({picked_easy} picked)"

            # rare + uncertain dominate the batch
            n_interesting = sum(1 for o in unc + rare if str(o) in ranks)
            assert n_interesting >= 3

            await db.delete(await db.get(DbSession, sid))
            await db.commit()
        return res

    out = asyncio.run(run())
    assert out["n_selected"] >= 3


@requires_infra
def test_al_loop_triggers_burst_finetune_on_force():
    from db.models import TrainingJob
    from db.session import get_sessionmaker
    from services.activelearn.loop import maybe_retrain

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            r = await maybe_retrain(db, compute_target="cloud", force=True)
            assert r["triggered"] is True and r["compute_target"] == "cloud"
            job = await db.get(TrainingJob, uuid.UUID(r["job_id"]))
            assert job is not None and job.purpose == "closed-loop" and job.compute_target == "cloud"
            await db.delete(job)
            await db.commit()

    asyncio.run(run())
