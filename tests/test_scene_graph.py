"""M-F.5 scene-graph relations + VLM dataset generation: geometry proposes each object's single best partner
per relation kind (no O(n^2) explosion on dense frames); a proposed relation is human-confirmed or rejected; and
a generated VLM target does not enter the export until a human approves it (the review gate). The VLM generation
call itself needs the model and is exercised operationally; here a target is seeded directly to test the gate.
Single asyncio.run so the cached engine binds to one loop."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.intelligence.scene_graph import SCENE_RELATIONS, propose_from_geometry, vocab


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def test_vocab_closed():
    v = vocab()
    assert set(v["relations"]) == set(SCENE_RELATIONS)
    assert "occluded_by" in v["geometric"] and "yielding_to" in v["vlm_or_human"]


def test_geometry_single_best_partner_no_explosion():
    # a vehicle behind another in the same lane (following), and a pedestrian ahead of the near vehicle
    # (crossing_in_front_of). A crowd of far tiny boxes must NOT each spawn a relation.
    W, H = 1920.0, 1080.0
    objs = [
        {"id": uuid.uuid4(), "bbox": [900, 700, 1020, 950], "l1": "four_wheeler"},   # near vehicle (low, big)
        {"id": uuid.uuid4(), "bbox": [910, 500, 1000, 620], "l1": "four_wheeler"},   # far vehicle ahead, same column
        {"id": uuid.uuid4(), "bbox": [905, 420, 960, 520], "l1": "vru"},             # pedestrian ahead of near vehicle
    ]
    # add 12 far tiny vehicles clustered together (would explode under pairwise emit)
    for k in range(12):
        x = 100 + k * 8
        objs.append({"id": uuid.uuid4(), "bbox": [x, 300, x + 12, 330], "l1": "four_wheeler"})
    props = propose_from_geometry(objs, W, H, cap=40)
    kinds = {p["kind"] for p in props}
    assert "following" in kinds
    # at most one relation of each kind FROM any single object (single-best-partner)
    from collections import Counter
    per_from_kind = Counter((str(p["from"]), p["kind"]) for p in props)
    assert all(v == 1 for v in per_from_kind.values())
    assert len(props) <= 40


@requires_infra
def test_vlm_target_review_gate_and_export():
    from sqlalchemy import delete, select
    from db.models import Frame, ObjectRelationship, Object, VlmTarget
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.vlm_dataset import export_dataset, set_target_status

    sid, fid = uuid.uuid4(), uuid.uuid4()

    async def run():
        maker = get_sessionmaker()
        async with maker() as db:
            db.add(DbSession(session_id=sid, vehicle_id="SG", start_ts_ns=0, end_ts_ns=1, ontology_version="labelox-in-0.1.0"))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=0, cam_id="cam_f", img_uri="s3://x/1.jpg", width=1920, height=1080))
            await db.flush()
            tgt = VlmTarget(frame_id=fid, session_id=sid, kind="scene_pack",
                            content={"scene_description": "a test street", "hazards": ["pedestrian"], "ego_action": {"action": "slow"}},
                            grounding={"object_ids": ["o1", "o2"], "track_ids": [], "relation_ids": ["r1"]},
                            model="qwen2.5vl:7b", status="generated")
            db.add(tgt)
            await db.commit()
            tid = tgt.target_id

        # review gate: a generated (unapproved) target does NOT export
        exp0 = await export_dataset(sid)
        assert exp0["n_samples"] == 0

        # approve -> it exports, traceable to its grounding
        await set_target_status(tid, "approved")
        exp1 = await export_dataset(sid)
        assert exp1["n_samples"] == 1
        assert exp1["format"] == "labelox-vlm-multimodal-v1"
        assert exp1["samples"][0]["grounding"]["relation_ids"] == ["r1"]

        async with maker() as db:
            await db.execute(delete(VlmTarget).where(VlmTarget.frame_id == fid))
            await db.delete(await db.get(DbSession, sid))
            await db.commit()

    asyncio.run(run())
