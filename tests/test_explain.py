"""M-F.0 explainable auto-labeling: the rationale is assembled from real provenance and replays the gate to
name the deciding reason. Covers an auto-accepted object (agreement + cleared floor), a demoted object (quality
flag), a single-path review, and a proposal with a null confidence (must not crash). Single asyncio.run so the
cached engine binds to one loop."""

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


@requires_infra
def test_explain_from_provenance():
    from sqlalchemy import delete
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.explain import explain_object

    sid = uuid.uuid4()

    def prov(**kw):
        base = {"proposals": [], "agreement": False, "mask_box_disagree": False, "quality_flags": []}
        base.update(kw)
        return base

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="EXPLAIN", start_ts_ns=0, end_ts_ns=1,
                             ontology_version="labelox-in-0.1.0"))
            await db.flush()
            f = Frame(session_id=sid, ts_ns=0, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080)
            db.add(f)
            await db.flush()

            # 1) auto-accept: two paths agree, high raw conf, no quality flags (class 19 = a benign vehicle)
            agreed = prov(agreement=True, calibrated_from=0.96, proposals=[
                {"path": "path_a_yolo26", "class_name": "truck", "conf": 0.96, "verdict": "agree"},
                {"path": "path_b_sam3", "class_name": "truck", "conf": 0.33, "verdict": "agree"}])
            # 2) quality-flagged -> demoted to review
            flagged = prov(agreement=True, calibrated_from=0.9, quality_flags=["impossible_size"], proposals=[
                {"path": "path_a_yolo26", "class_name": "sedan", "conf": 0.9, "verdict": "agree"}])
            # 3) single path, no agreement -> review
            single = prov(agreement=False, calibrated_from=0.5, proposals=[
                {"path": "path_a_yolo26", "class_name": "sedan", "conf": 0.5, "verdict": "agree"}])
            # 4) a proposal with a NULL conf must not crash the explainer
            nullconf = prov(agreement=False, calibrated_from=0.5, proposals=[
                {"path": "path_a_yolo26", "class_name": "sedan", "conf": None, "verdict": "agree"},
                {"path": "path_b_sam3", "class_name": "pickup", "conf": None, "verdict": "overruled"}])
            objs = {}
            for name, p, cid in (("agreed", agreed, 19), ("flagged", flagged, 11), ("single", single, 11),
                                 ("nullconf", nullconf, 11)):
                o = Object(frame_id=f.frame_id, class_id=cid, bbox=[10, 10, 60, 60], conf=0.5,
                           state="review", source="fused", provenance=p)
                db.add(o)
                await db.flush()
                objs[name] = o.object_id
            await db.commit()

        async with maker() as db:
            e = await explain_object(db, objs["agreed"])
            assert e["machine_decision"] == "auto_accept"
            assert e["agreement"] is True and len(e["paths"]) == 2
            assert any("agreed on the class" in s for s in e["summary"])

            e = await explain_object(db, objs["flagged"])
            assert e["machine_decision"] == "review"
            assert e["quality_flags"] == ["impossible_size"]
            assert any("quality reviewer" in s.lower() for s in e["summary"])

            e = await explain_object(db, objs["single"])
            assert e["machine_decision"] == "review"

            # the null-conf object explains without crashing and reports 0.0 for the missing scores
            e = await explain_object(db, objs["nullconf"])
            assert e["object_id"] == str(objs["nullconf"])
            assert all(p["conf"] == 0.0 for p in e["paths"])
            assert "pickup" in e["overruled_classes"]

        async with maker() as db:
            await db.delete(await db.get(DbSession, sid))  # cascades frames/objects
            await db.commit()

    asyncio.run(run())
