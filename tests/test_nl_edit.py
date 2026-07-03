"""M-F.3 natural-language bulk editing: a command parses into an operation + selection, resolves to concrete
objects (preview only, no mutation), and applies to the confirmed objects as ONE reversible, audited agent run
that routes them to review. select never mutates; reclassify without a valid target is refused. Single
asyncio.run so the cached engine binds to one loop."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.agent.nl_edit import parse_edit


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def test_parse_operations_and_filters():
    onto = get_ontology()
    p1 = parse_edit("select every autorickshaw", onto)
    assert p1["operation"] == "select" and p1["select_class_ids"]
    p2 = parse_edit("reclassify the fallback objects to push_cart", onto)
    assert p2["operation"] == "reclassify" and p2["to_class_id"] is not None and p2["select_class_ids"]
    p3 = parse_edit("flag every pedestrian with no visible mask", onto)
    assert p3["operation"] == "flag" and p3["mask_missing"] is True
    p4 = parse_edit("select every parked autorickshaw near the barrier", onto)
    assert p4["attr"] and p4["attr"]["name"] == "motion" and p4["referential"] is True


@requires_infra
def test_preview_apply_and_revert():
    from sqlalchemy import select
    from db.models import AgentRun, AuditDecision, Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.agent.nl_edit import apply_edit, resolve
    from services.agent.runs import revert_run

    onto = get_ontology()
    fb_ids = onto.fallback_ids()
    assert fb_ids, "ontology must have a fallback class for this test"
    src_cls = fb_ids[0]
    to_cls = onto.by_name("push_cart").id if onto.has_name("push_cart") else [c.id for c in onto.classes if c.id != src_cls][0]
    sid, fid = uuid.uuid4(), uuid.uuid4()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="NLEDIT", start_ts_ns=0, end_ts_ns=1, ontology_version=onto.version))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=0, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080))
            await db.flush()
            oids = []
            for _ in range(4):
                o = Object(frame_id=fid, class_id=src_cls, bbox=[10, 10, 60, 60], conf=0.5, state="auto_accept", source="fused")
                db.add(o)
                await db.flush()
                oids.append(o.object_id)
            await db.commit()

        plan = parse_edit("reclassify the fallback objects to push_cart", onto)
        plan["to_class_id"] = to_cls  # pin the target for the test's ontology
        plan["select_class_ids"] = [src_cls]

        async with maker() as db:
            preview = await resolve(db, plan, frame_id=fid)
            assert preview["count"] == 4
            # select is preview-only: apply refuses it
            sel = await apply_edit(db, {**plan, "operation": "select"}, [uuid.UUID(o["object_id"]) for o in preview["objects"]])
            assert "error" in sel

            r = await apply_edit(db, plan, [uuid.UUID(o["object_id"]) for o in preview["objects"]], created_by=None)
            assert r["edited"] == 4 and r["routed_to"] == "review"

        async with maker() as db:
            objs = (await db.execute(select(Object).where(Object.object_id.in_(oids)))).scalars().all()
            assert all(o.class_id == to_cls and o.state == "review" for o in objs)
            assert all(o.provenance.get("agent_run_id") == r["run_id"] for o in objs)
            assert await db.get(AgentRun, uuid.UUID(r["run_id"])) is not None
            aud = (await db.execute(select(AuditDecision).where(AuditDecision.subject == r["run_id"]))).scalar_one_or_none()
            assert aud is not None and aud.actor == "nl_edit"

            rv = await revert_run(db, uuid.UUID(r["run_id"]))
            assert rv["reverted"] == 4

        async with maker() as db:
            objs = (await db.execute(select(Object).where(Object.object_id.in_(oids)))).scalars().all()
            assert all(o.class_id == src_cls for o in objs)  # restored
            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
