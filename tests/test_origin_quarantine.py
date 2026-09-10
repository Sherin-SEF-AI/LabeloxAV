"""The synthetic quarantine, proven on rows: nothing that measures a model can see a composite.

One real session and one synthetic session are seeded with the same shape of labels. Every reader below is
then asked the question it answers in production and must return only the real rows. Each assertion pairs
with a positive one on the real session, so a reader that silently returned nothing at all would fail the
test rather than pass it by accident.

Two mechanisms are under test and they are different in kind. Allow-list readers (settlement population,
the control-sample seed, the precision draw, gold) select by object state and never name `synthetic`, so the
generator's own state keeps them clean. Deny-list readers (the trainset, embedding, the explorer, the blind
audit) select every state and must say `Frame.origin == REAL` themselves; those are the ones this test
exists for.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select, text

from core.origin import REAL, SYNTHETIC, SYNTHETIC_SOURCE, SYNTHETIC_STATE
from core.timebase import now_ns
from db.models import (
    ControlSample,
    Frame,
    InferenceRun,
    LabelProject,
    ModelRegistry,
    Object,
    Prediction,
)
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

EPOCH = "quarantine-model-A"
CLASS = "pedestrian"
STATES = ("review", "auto_accept", "accepted", "settled")


async def _seed(db) -> dict:
    """A real session and a synthetic one, four frames each, one object per state per frame."""
    onto = get_ontology()
    cid = onto.by_name(CLASS).id
    tag = uuid.uuid4().hex[:6]
    out: dict = {"cid": cid, "frames": {REAL: [], SYNTHETIC: []}, "objects": {REAL: [], SYNTHETIC: []}}
    real_sid = uuid.uuid4()
    synth_sid = uuid.uuid4()
    for origin, sid in ((REAL, real_sid), (SYNTHETIC, synth_sid)):
        db.add(DbSession(session_id=sid, vehicle_id=f"Q-{tag}", start_ts_ns=0, end_ts_ns=1, city="BLR",
                         route=f"quarantine-{tag}", sensors={}, ontology_version=onto.version, origin=origin))
        out[f"{origin}_session"] = sid
    await db.flush()
    prov = {"proposals": [{"verdict": "agree", "class_name": CLASS, "model_version": EPOCH,
                           "conf": 0.8, "path": "path_a"}]}
    for i in range(4):
        rf = Frame(frame_id=uuid.uuid4(), session_id=real_sid, ts_ns=now_ns() + i, cam_id="front",
                   width=1920, height=1080, img_uri=f"s3://x/{tag}-real-{i}.jpg", origin=REAL)
        db.add(rf)
        out["frames"][REAL].append(rf.frame_id)
        sf = Frame(frame_id=uuid.uuid4(), session_id=synth_sid, ts_ns=now_ns() + i, cam_id="front",
                   width=1920, height=1080, img_uri=f"s3://x/{tag}-synth-{i}.jpg", origin=SYNTHETIC,
                   source_frame_id=rf.frame_id)
        db.add(sf)
        out["frames"][SYNTHETIC].append(sf.frame_id)
        await db.flush()
        for st in STATES:
            src = "human" if st == "accepted" else "fused"
            o = Object(frame_id=rf.frame_id, class_id=cid, bbox=[100, 100, 400, 400], conf=0.9,
                       source=src, state=st, provenance=prov, attrs={})
            db.add(o)
            await db.flush()
            out["objects"][REAL].append(o.object_id)
        # The generator writes exactly this: state and source synthetic, provenance naming its sources.
        for _ in STATES:
            o = Object(frame_id=sf.frame_id, class_id=cid, bbox=[100, 100, 400, 400], conf=0.9,
                       source=SYNTHETIC_SOURCE, state=SYNTHETIC_STATE,
                       provenance={**prov, "synthetic": {"source_frame_id": str(rf.frame_id), "pasted": True}},
                       attrs={})
            db.add(o)
            await db.flush()
            out["objects"][SYNTHETIC].append(o.object_id)
    await db.commit()
    return out


class TestSchema:
    async def test_origin_checks_and_extended_state_check_exist(self):
        async with get_sessionmaker()() as db:
            rows = (await db.execute(text(
                "select conname, pg_get_constraintdef(oid) from pg_constraint "
                "where conname in ('ck_session_origin','ck_frame_origin','ck_object_state','ck_object_source')"))).all()
        defs = {name: d for name, d in rows}
        assert set(defs) == {"ck_session_origin", "ck_frame_origin", "ck_object_state", "ck_object_source"}
        for name in ("ck_session_origin", "ck_frame_origin"):
            assert "'real'" in defs[name] and "'synthetic'" in defs[name] and "'perturbed'" in defs[name]
        assert "'synthetic'" in defs["ck_object_state"]
        assert "'synthetic'" in defs["ck_object_source"]

    async def test_session_and_frame_origin_agree(self):
        """`Session.origin` is for listing and `Frame.origin` is the working predicate; they never disagree."""
        async with get_sessionmaker()() as db:
            await _seed(db)
            n = (await db.execute(
                select(func.count()).select_from(Frame).join(DbSession, DbSession.session_id == Frame.session_id)
                .where(Frame.origin != DbSession.origin))).scalar_one()
            assert n == 0

    def test_a_person_cannot_make_an_object_synthetic(self):
        from services.review_policy import OBJECT_STATES, ReviewStateError, state_for

        assert SYNTHETIC_STATE in OBJECT_STATES
        with pytest.raises(ReviewStateError, match="synthetic"):
            state_for(None, SYNTHETIC_STATE, "admin", "review")


class TestAllowListReaders:
    async def test_settlement_population(self):
        from services.labelops.settlement import stratum_population

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            before = await stratum_population(db, fx["cid"], CLASS, EPOCH)
        # The real session added four review objects under EPOCH; the synthetic one added none.
        async with get_sessionmaker()() as db:
            real_review = (await db.execute(select(func.count()).select_from(Object).join(Frame)
                                            .where(Frame.origin == REAL, Object.state == "review",
                                                   Object.class_id == fx["cid"]))).scalar_one()
        assert before >= 4 and before <= real_review

    async def test_control_sample_seed(self):
        from services.govern.control_sample import seed_from_recent_auto_accepts

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            await seed_from_recent_auto_accepts(db, limit=100_000, rate=1.0)
            mirrored = set((await db.execute(select(ControlSample.object_id))).scalars().all())
        synth = set(fx["objects"][SYNTHETIC])
        assert not (mirrored & synth)
        real_auto = [o for o in fx["objects"][REAL]]
        assert mirrored & set(real_auto)

    async def test_precision_draw(self):
        from services.labelops.precision_batch import build_precision_batch

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            res = await build_precision_batch(db, classes=[CLASS], target=100_000, min_side_px=1)
            stamped = set((await db.execute(select(Object.object_id).where(
                Object.provenance["flywheel"]["cycle_id"].astext == res["batch_id"]))).scalars().all())
        assert stamped, "the draw found nothing at all; the positive half of this test is void"
        assert not (stamped & set(fx["objects"][SYNTHETIC]))

    async def test_gold_build(self):
        from services.training.gold import GoldSpec, _fetch_gold

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
        real = await _fetch_gold(GoldSpec(session_id=str(fx[f"{REAL}_session"])))
        synth = await _fetch_gold(GoldSpec(session_id=str(fx[f"{SYNTHETIC}_session"])))
        assert len(real) == 4
        assert synth == []


class TestDenyListReaders:
    async def test_trainset_select_and_val_split(self):
        from services.training.dataset_builder import BuildSpec, _partition, _select

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
        route = None
        async with get_sessionmaker()() as db:
            route = (await db.get(DbSession, fx[f"{REAL}_session"])).route
        real_ids = {str(f) for f in fx["frames"][REAL]}
        synth_ids = {str(f) for f in fx["frames"][SYNTHETIC]}

        off = await _select(BuildSpec(route_prefix=route, conf_floor=0.0))
        assert {c["frame_id"] for c in off} == real_ids

        on = await _select(BuildSpec(route_prefix=route, conf_floor=0.0, include_synthetic=True))
        assert {c["frame_id"] for c in on} == real_ids | synth_ids
        assert all(c["origin"] == SYNTHETIC and c["source_frame_id"] in real_ids
                   for c in on if c["frame_id"] in synth_ids)

        # Validation never holds a composite, and a composite whose background is in val is dropped.
        by_frame: dict[str, list[dict]] = {}
        for c in on:
            by_frame.setdefault(c["frame_id"], []).append(c)
        # Add a second real session's worth of frames so a session-grouped split has something to choose.
        for i in range(8):
            fid = f"extra-{i}"
            by_frame[fid] = [{"frame_id": fid, "session_id": f"extra-session-{i % 2}", "origin": REAL,
                              "source_frame_id": None, "gold": False, "class_id": fx["cid"]}]
        spec = BuildSpec(val_frac=0.5, seed=1)
        val, kept, dropped = _partition(by_frame, set(), spec)
        assert not (val & synth_ids)
        val_sessions = {by_frame[f][0]["session_id"] for f in val}
        for sid in synth_ids:
            src = str(fx["frames"][REAL][list(map(str, fx["frames"][SYNTHETIC])).index(sid)])
            src_in_val = src in val or str(fx[f"{REAL}_session"]) in val_sessions
            assert (sid in by_frame) == (not src_in_val)
        assert kept + dropped == 4

    async def test_embedding_queue(self):
        from services.intelligence.embed.pending import (
            frame_needs_embedding,
            object_needs_embedding,
        )

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            synth_frames = fx["frames"][SYNTHETIC]
            synth_objs = fx["objects"][SYNTHETIC]
            f_pending = (await db.execute(select(func.count()).select_from(Frame).where(
                Frame.frame_id.in_(synth_frames), frame_needs_embedding()))).scalar_one()
            o_pending = (await db.execute(select(func.count()).select_from(Object).where(
                Object.object_id.in_(synth_objs), object_needs_embedding()))).scalar_one()
            f_real = (await db.execute(select(func.count()).select_from(Frame).where(
                Frame.frame_id.in_(fx["frames"][REAL]), frame_needs_embedding()))).scalar_one()
        assert f_pending == 0 and o_pending == 0
        assert f_real == 4

    async def test_explorer_predicate(self):
        from services.explore.query import frame_clauses

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            sid = fx[f"{SYNTHETIC}_session"]
            hidden = (await db.execute(select(func.count()).select_from(Frame)
                                       .where(*frame_clauses({"session_id": str(sid)})))).scalar_one()
            shown = (await db.execute(select(func.count()).select_from(Frame)
                                      .where(*frame_clauses({"session_id": str(sid),
                                                             "include_synthetic": True})))).scalar_one()
        assert hidden == 0 and shown == 4

    async def test_blind_audit_never_picks_a_composite(self):
        from services.verdyx.blind_audit import audit_frame_ids, seed_audit

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
            mv = f"quarantine-{uuid.uuid4().hex[:8]}"
            db.add(ModelRegistry(model_version=mv, task="detection"))
            await db.flush()
            run = InferenceRun(model_version=mv, gold_id=None, status="complete", frame_count=8, params={},
                               code_sha="0" * 40)
            db.add(run)
            await db.flush()
            for fid in fx["frames"][REAL] + fx["frames"][SYNTHETIC]:
                db.add(Prediction(run_id=run.run_id, frame_id=fid, class_id=fx["cid"],
                                  bbox=[0.0, 0.0, 100.0, 100.0], conf=0.9))
            project = LabelProject(name=f"quarantine-{uuid.uuid4().hex[:6]}", modality="image")
            db.add(project)
            await db.commit()
            res = await seed_audit(db, run_id=str(run.run_id), n_frames=100, project_id=str(project.project_id))
            assert "error" not in res, res
            picked = set(await audit_frame_ids(db, uuid.UUID(res["audit_id"])))
        assert picked == set(fx["frames"][REAL])


class TestMigrationDowngrade:
    async def test_downgrade_leaves_no_synthetic_rows(self):
        """0108's downgrade must remove every composite before it tightens the CHECKs back, or the
        constraint re-creation fails and leaves the schema half way. Runs the real alembic scripts."""
        import subprocess

        async with get_sessionmaker()() as db:
            fx = await _seed(db)
        env = {"LBX_POSTGRES__DB": "labeloxav_test"}
        import os

        full = {**os.environ, **env}
        down = subprocess.run([".venv/bin/alembic", "downgrade", "0107_settlement_sprt"], env=full,
                              capture_output=True, text=True)
        assert down.returncode == 0, down.stderr[-2000:]
        try:
            async with get_sessionmaker()() as db:
                gone = (await db.execute(text("select count(*) from session where session_id = :s"),
                                         {"s": str(fx[f"{SYNTHETIC}_session"])})).scalar_one()
                kept = (await db.execute(text("select count(*) from session where session_id = :s"),
                                         {"s": str(fx[f"{REAL}_session"])})).scalar_one()
                cols = (await db.execute(text(
                    "select column_name from information_schema.columns where table_name='frame' "
                    "and column_name in ('origin','source_frame_id')"))).scalars().all()
            assert gone == 0 and kept == 1
            assert cols == []
        finally:
            up = subprocess.run([".venv/bin/alembic", "upgrade", "head"], env=full, capture_output=True, text=True)
            assert up.returncode == 0, up.stderr[-2000:]
