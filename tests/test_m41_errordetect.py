"""M4.1 annotation error detection: seeded label errors in accepted data are surfaced and ranked, a
cross-camera inconsistency is detected, and confirming an error feeds it into the correction + retrain
path (a Review row + confirmed_error status the loop counts). Single asyncio.run."""

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
def test_errordetect_confident_outlier_crosscam_and_feedback():
    from db.models import ErrorCandidate, Frame, Object, ObjectEmbedding, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.errordetect.queue import confirm_error, run_detection

    sid = uuid.uuid4()

    def cluster_vec(center_seed, jitter_seed):
        c = np.random.default_rng(center_seed).standard_normal(768).astype("float32")
        j = np.random.default_rng(jitter_seed).standard_normal(768).astype("float32") * 0.05
        v = c + j
        return (v / np.linalg.norm(v)).tolist()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="ED-TEST", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=640, height=480)
            db.add(f)
            await db.flush()

            # class 1 cluster (10 objects near center A), class 2 cluster (10 near center B), both accepted
            mislabeled = None
            for k in range(10):
                o = Object(frame_id=f.frame_id, class_id=1, bbox=[k, 0, k + 5, 5], conf=0.9,
                           state="accepted", source="human")
                db.add(o); await db.flush()
                db.add(ObjectEmbedding(object_id=o.object_id, dino_vec=cluster_vec(1, 1000 + k), model_versions={}))
            for k in range(10):
                o = Object(frame_id=f.frame_id, class_id=2, bbox=[k, 10, k + 5, 15], conf=0.9,
                           state="accepted", source="human")
                db.add(o); await db.flush()
                db.add(ObjectEmbedding(object_id=o.object_id, dino_vec=cluster_vec(2, 2000 + k), model_versions={}))
            # the seeded error: labeled class 1 but its embedding sits squarely in cluster B
            bad = Object(frame_id=f.frame_id, class_id=1, bbox=[0, 20, 5, 25], conf=0.9,
                         state="auto_accept", source="auto_accept")
            db.add(bad); await db.flush()
            mislabeled = bad.object_id
            db.add(ObjectEmbedding(object_id=bad.object_id, dino_vec=cluster_vec(2, 2999), model_versions={}))

            # a cross-camera inconsistency: one rig identity labeled two different classes across views
            rid = uuid.uuid4()
            ca = Object(frame_id=f.frame_id, class_id=1, bbox=[100, 0, 110, 10], conf=0.9,
                        state="accepted", source="human", rig_track_id=rid)
            cb = Object(frame_id=f.frame_id, class_id=2, bbox=[100, 0, 110, 10], conf=0.9,
                        state="accepted", source="human", rig_track_id=rid)
            cc = Object(frame_id=f.frame_id, class_id=1, bbox=[100, 0, 110, 10], conf=0.9,
                        state="accepted", source="human", rig_track_id=rid)
            db.add_all([ca, cb, cc]); await db.flush()
            crosscam_minority = cb.object_id  # the lone class-2 view among class-1 majority
            await db.commit()

            res = await run_detection(db, session_id=str(sid))
            assert res["persisted"] >= 2

            cands = (await db.execute(ErrorCandidate.__table__.select().where(
                ErrorCandidate.object_id.in_([mislabeled, crosscam_minority])))).all()
            kinds = {row.object_id: row.kind for row in cands}
            # the embedding-misplaced object is caught (confident_learning or embedding_outlier)
            assert mislabeled in kinds, "seeded label error not surfaced"
            # the cross-camera minority is caught
            assert crosscam_minority in kinds and kinds[crosscam_minority] == "cross_cam_inconsistent"

            # confirming the seeded error applies the fix and writes a correction the retrain counts
            cand_id = next(str(row.candidate_id) for row in cands if row.object_id == mislabeled)
            verdict = await confirm_error(db, cand_id, apply_proposed=True)
            assert verdict["status"] == "confirmed_error"
            n_rev = (await db.execute(Review.__table__.select().where(Review.object_id == mislabeled))).all()
            assert len(n_rev) == 1  # a correction landed
            cc_row = await db.get(ErrorCandidate, uuid.UUID(cand_id))
            assert cc_row.status == "confirmed_error"

            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
