"""M4.2 AI-assisted relabeling: an ontology-promotion relabel run re-classifies vehicle_fallback objects
to a named class, auto-applies safe improvements WITHOUT touching human-verified objects, routes the
human conflict to review, lands on a new lakeFS branch, and is fully reversible. The diff classifier is a
pure unit. Single asyncio.run for the DB path."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.relabel.diff import classify_change


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def test_diff_classifier_protects_human_and_flags_regression():
    onto = get_ontology()
    fb, named = onto.by_name("vehicle_fallback").id, onto.by_name("motorcycle").id
    base = {"old_class_id": fb, "old_class": "vehicle_fallback", "new_class_id": named, "new_class": "motorcycle"}
    # ontology promotion of a non-human object auto-applies
    assert classify_change({**base, "source": "fused", "old_conf": 0.7, "new_conf": 0.7,
                            "reason": "ontology_promotion"})["apply"] is True
    # the same promotion on a human-verified object is a conflict, never auto-applied
    assert classify_change({**base, "source": "human", "old_conf": 0.7, "new_conf": 0.7,
                            "reason": "ontology_promotion"})["verdict"] == "conflict"
    # a model change that drops confidence is a regression, routed not applied
    reg = classify_change({**base, "source": "fused", "old_conf": 0.9, "new_conf": 0.6, "reason": "model_reinfer"})
    assert reg["verdict"] == "regression" and reg["apply"] is False


@requires_infra
def test_relabel_promotes_applies_safely_and_reverts():
    from db.models import Frame, Object, RelabelRun
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.relabel.apply import revert_run
    from services.relabel.run import run_relabel_job, start_relabel
    from db.models import RelabelJob

    sid = uuid.uuid4()
    onto = get_ontology()
    fb_id = onto.by_name("vehicle_fallback").id

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="RL-TEST", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f = Frame(session_id=sid, ts_ns=1000, cam_id="cam_f", img_uri="s3://x/1.jpg", width=640, height=480)
            db.add(f)
            await db.flush()
            # 3 auto-accepted fallback objects (safe to promote) + 1 human-verified fallback (must NOT change)
            auto_ids, human_id = [], None
            for k in range(3):
                o = Object(frame_id=f.frame_id, class_id=fb_id, bbox=[k, 0, k + 5, 5], conf=0.8,
                           state="auto_accept", source="auto_accept")
                db.add(o); await db.flush(); auto_ids.append(o.object_id)
            h = Object(frame_id=f.frame_id, class_id=fb_id, bbox=[50, 0, 55, 5], conf=0.95,
                       state="accepted", source="human")
            db.add(h); await db.flush(); human_id = h.object_id
            await db.commit()

        # start a local ontology-promotion relabel: vehicle_fallback -> motorcycle, scoped to this session
        res = await start_relabel("champion-v1", session_ids=[str(sid)],
                                  ontology_promotion={"from_class": "vehicle_fallback", "to_class": "motorcycle"},
                                  compute_target="local")
        assert res["compute_target"] == "local"
        assert res["applied"] == 3 and res["conflicts"] == 1  # 3 auto promoted, 1 human protected
        assert res["branch"].startswith("relabel-motorcycle")
        assert res["commit"], "no lakeFS commit"
        run_id = res["run_id"]

        moto_id = onto.by_name("motorcycle").id
        async with maker() as db:
            for oid in auto_ids:
                o = await db.get(Object, oid)
                assert o.class_id == moto_id and o.source == "relabel"  # promoted
            hobj = await db.get(Object, human_id)
            assert hobj.class_id == fb_id and hobj.source == "human"   # human untouched
            run_row = (await db.execute(RelabelRun.__table__.select().where(RelabelRun.run_id == uuid.UUID(run_id)))).first()
            assert run_row is not None

        # reversible: revert restores the original fallback class on the promoted objects
        async with maker() as db:
            rev = await revert_run(db, run_id)
            assert rev["restored"] == 3
        async with maker() as db:
            for oid in auto_ids:
                o = await db.get(Object, oid)
                assert o.class_id == fb_id  # restored
            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
