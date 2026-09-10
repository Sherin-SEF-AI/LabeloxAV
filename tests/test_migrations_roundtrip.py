"""Every migration from 0107 on, down and back up, on the test database, with rows in the way.

A migration whose `downgrade()` has never run is a one-way door dressed as a two-way one. This walks
the whole post-settlement range (`FLOOR` to head) down and up again with a seeded row on each table
the range touches, and asserts that the row survives, that the downgraded schema has forgotten the
columns, and that the re-upgraded row carries the server defaults the upgrade promised. New
migrations extend `_seed` and `_assert_downgraded`; the walk itself does not change.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

from core.config import get_settings

pytestmark = pytest.mark.db

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOOR = "0106_settlement"       # the revision the walk descends to; everything above it round-trips


def _alembic(*args: str) -> None:
    subprocess.run([".venv/bin/alembic", *args], check=True, cwd=REPO_ROOT, env={**os.environ},
                   capture_output=True)


def _current() -> str:
    out = subprocess.run([".venv/bin/alembic", "current"], check=True, cwd=REPO_ROOT,
                         env={**os.environ}, capture_output=True, text=True).stdout
    return out.strip().split()[0] if out.strip() else ""


def _columns(conn, table: str) -> set[str]:
    return {r[0] for r in conn.execute(sa.text(
        "select column_name from information_schema.columns where table_name = :t"), {"t": table})}


def _seed(conn) -> dict:
    """One row on every table the 0107+ range touches, with the non-default values the range adds."""
    from services.autolabel.ontology import get_ontology

    lot_id = uuid.uuid4()
    cid = get_ontology().by_name("sedan").id
    conn.execute(sa.text("""
        insert into settlement_lot (lot_id, class_id, model_epoch, population, tier, far_bound,
            sample_object_ids, batch_id, status, created_by, rule, cap_n, llr, sprt, increments)
        values (:lot, :cid, 'roundtrip-epoch', 5000, 'default', 0.05, '[]'::jsonb, :bid, 'judging',
                'roundtrip', 'sprt', 110, -1.25, '{"p0": 0.05}'::jsonb, '[{"added": 25}]'::jsonb)"""),
        {"lot": lot_id, "cid": cid, "bid": f"settle-{lot_id.hex[:8]}"})
    # 0108: a synthetic session with a composite frame and a synthetic-state object, plus a real session
    # the composite points back at. The downgrade must remove the synthetic rows (the tighter CHECKs it
    # restores would otherwise refuse to come back) and keep the real ones.
    real_sid, synth_sid = uuid.uuid4(), uuid.uuid4()
    real_fid, synth_fid = uuid.uuid4(), uuid.uuid4()
    for sid, origin in ((real_sid, "real"), (synth_sid, "synthetic")):
        conn.execute(sa.text("""
            insert into session (session_id, vehicle_id, start_ts_ns, end_ts_ns, sensors, ontology_version, origin)
            values (:sid, 'roundtrip', 0, 1, '{}'::jsonb, :ov, :origin)"""),
            {"sid": sid, "ov": get_ontology().version, "origin": origin})
    conn.execute(sa.text("""
        insert into frame (frame_id, session_id, ts_ns, cam_id, img_uri, width, height, quality, origin)
        values (:fid, :sid, 0, 'front', 's3://x/roundtrip-real.jpg', 10, 10, 0, 'real')"""),
        {"fid": real_fid, "sid": real_sid})
    conn.execute(sa.text("""
        insert into frame (frame_id, session_id, ts_ns, cam_id, img_uri, width, height, quality, origin,
                           source_frame_id)
        values (:fid, :sid, 0, 'front', 's3://x/roundtrip-synth.jpg', 10, 10, 0, 'synthetic', :src)"""),
        {"fid": synth_fid, "sid": synth_sid, "src": real_fid})
    conn.execute(sa.text("""
        insert into object (object_id, frame_id, class_id, bbox, conf, source, state, provenance, attrs)
        values (:oid, :fid, :cid, '{0,0,5,5}', 1.0, 'synthetic', 'synthetic', '{}'::jsonb, '{}'::jsonb)"""),
        {"oid": uuid.uuid4(), "fid": synth_fid, "cid": cid})
    # 0109/0110: a disagreement needs a sweep run, two inference runs and a registered model to hang on,
    # and 0110's lineage columns need a row carrying non-default values or the downgrade would be
    # dropping columns nothing ever wrote to.
    sweep_id = uuid.uuid4()
    conn.execute(sa.text("""
        insert into agent_run (run_id, kind, scope, status, policy, counts, changes, critic, created_by)
        values (:r, 'shadow_sweep', '{}'::jsonb, 'committed', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
                '{}'::jsonb, 'roundtrip')"""), {"r": sweep_id})
    mv = f"roundtrip-model-{sweep_id.hex[:8]}"
    conn.execute(sa.text("""
        insert into model_registry (model_version, task, gold_metrics, is_champion, arch, parent_version,
                                    teacher_version, origin)
        values (:mv, 'detection', '{}'::jsonb, false, 'yolo11n', 'yolo11n.pt', :mv2, 'distilled')"""),
        {"mv": mv, "mv2": f"{mv}-teacher"})
    run_ids = []
    for _ in range(2):
        rid = uuid.uuid4()
        conn.execute(sa.text("""
            insert into inference_run (run_id, model_version, params, code_sha, status, frame_count)
            values (:r, :mv, '{}'::jsonb, 'roundtrip', 'complete', 1)"""), {"r": rid, "mv": mv})
        run_ids.append(rid)
    dis_id = uuid.uuid4()
    pred_id = uuid.uuid4()
    conn.execute(sa.text("""
        insert into prediction (prediction_id, run_id, frame_id, class_id, bbox, conf)
        values (:p, :r, :f, :cid, '{0,0,5,5}', 0.9)"""),
        {"p": pred_id, "r": run_ids[0], "f": real_fid, "cid": cid})
    conn.execute(sa.text("""
        insert into shadow_disagreement (disagreement_id, sweep_run_id, frame_id, champion_run_id,
            challenger_run_id, champion_prediction_id, kind, champion_class_id, score, bbox, state)
        values (:d, :s, :f, :cr, :xr, :p, 'challenger_miss', :cid, 0.9, '{0,0,5,5}', 'pending')"""),
        {"d": dis_id, "s": sweep_id, "f": real_fid, "cr": run_ids[0], "xr": run_ids[1],
         "p": pred_id, "cid": cid})
    # 0111/0112: a run with a declared class vocabulary, and an ego pose on the real frame. The pose
    # carries a real quaternion because the table checks that it is a unit one.
    conn.execute(sa.text("update inference_run set class_vocab = :v where run_id = :r"),
                 {"v": '[1, 2, 3]', "r": run_ids[0]})
    conn.execute(sa.text("""
        insert into ego_pose (session_id, ts_ns, frame_id, x, y, z, qw, qx, qy, qz, speed_mps,
                              source, quality, measured)
        values (:s, 0, :f, 1.5, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0, 8.3, 'visual', 0.42, false)"""),
        {"s": real_sid, "f": real_fid})
    # 0113: an occupancy grid on the real frame, with a flow count inside its occupied count so the
    # CHECK the migration adds is exercised rather than merely present.
    grid_id = uuid.uuid4()
    conn.execute(sa.text("""
        insert into occupancy_grid (grid_id, session_id, ts_ns, frame_id, origin, voxel_m, dims,
                                    grid_uri, source, ego_pose_ts, occupied, flow_voxels)
        values (:g, :s, 0, :f, '{0,0,0}', 0.5, '{10,10,10}', 's3://x/grid.npz', 'pseudo', 0, 120, 30)"""),
        {"g": grid_id, "s": real_sid, "f": real_fid})
    return {"lot_id": lot_id, "real_sid": real_sid, "synth_sid": synth_sid,
            "sweep_id": sweep_id, "model_version": mv, "disagreement_id": dis_id,
            "vocab_run_id": run_ids[0], "grid_id": grid_id}


def _assert_downgraded(conn, seeded: dict) -> None:
    cols = _columns(conn, "settlement_lot")
    for c in ("rule", "cap_n", "llr", "sprt", "increments"):
        assert c not in cols, f"0107 downgrade left settlement_lot.{c}"
    assert conn.execute(sa.text("select count(*) from settlement_lot where lot_id = :l"),
                        {"l": seeded["lot_id"]}).scalar() == 1, "the downgrade must keep the lot"
    fcols = _columns(conn, "frame")
    assert "origin" not in fcols and "source_frame_id" not in fcols, "0108 downgrade left frame.origin"
    assert "origin" not in _columns(conn, "session")
    assert conn.execute(sa.text("select count(*) from session where session_id = :s"),
                        {"s": seeded["synth_sid"]}).scalar() == 0, "0108 downgrade must remove the composite"
    assert conn.execute(sa.text("select count(*) from session where session_id = :s"),
                        {"s": seeded["real_sid"]}).scalar() == 1, "0108 downgrade must keep the real session"
    assert conn.execute(sa.text("select count(*) from object where state = 'synthetic'")).scalar() == 0
    assert conn.execute(sa.text(
        "select to_regclass('public.shadow_disagreement')")).scalar() is None, \
        "0109 downgrade left the shadow_disagreement table"
    mcols = _columns(conn, "model_registry")
    for c in ("arch", "parent_version", "teacher_version", "origin"):
        assert c not in mcols, f"0110 downgrade left model_registry.{c}"
    assert conn.execute(sa.text("select count(*) from model_registry where model_version = :m"),
                        {"m": seeded["model_version"]}).scalar() == 1, \
        "0110 drops columns, never the model rows that carried them"
    assert "class_vocab" not in _columns(conn, "inference_run"), \
        "0111 downgrade left inference_run.class_vocab"
    assert conn.execute(sa.text("select count(*) from inference_run where run_id = :r"),
                        {"r": seeded["vocab_run_id"]}).scalar() == 1, \
        "0111 drops a column, never the runs that carried it"
    assert conn.execute(sa.text("select to_regclass('public.ego_pose')")).scalar() is None, \
        "0112 downgrade left the ego_pose table"
    assert conn.execute(sa.text("select to_regclass('public.occupancy_grid')")).scalar() is None, \
        "0113 downgrade left the occupancy_grid table"


def _assert_reupgraded(conn, seeded: dict) -> None:
    row = conn.execute(sa.text(
        "select rule, cap_n, llr, sprt, increments from settlement_lot where lot_id = :l"),
        {"l": seeded["lot_id"]}).one()
    assert row.rule == "wilson" and row.cap_n == 0 and row.llr is None, \
        "a lot that lived through the downgrade is a fixed-rule lot: the stricter reading"
    assert row.sprt == {} and row.increments == []
    assert conn.execute(sa.text("select origin from session where session_id = :s"),
                        {"s": seeded["real_sid"]}).scalar() == "real"
    # The disagreements are gone with the table and the lineage is back to its default: a downgrade that
    # dropped them cannot invent them again, and claiming otherwise would be the lie this test exists for.
    assert conn.execute(sa.text("select count(*) from shadow_disagreement")).scalar() == 0
    assert conn.execute(sa.text("select origin from model_registry where model_version = :m"),
                        {"m": seeded["model_version"]}).scalar() == "trained"
    # A vocabulary the downgrade dropped is gone, and null is the honest reading of that: this run
    # declared nothing, which the matcher reads as "compare every class" rather than as an empty one.
    assert conn.execute(sa.text("select class_vocab from inference_run where run_id = :r"),
                        {"r": seeded["vocab_run_id"]}).scalar() is None
    assert conn.execute(sa.text("select count(*) from ego_pose")).scalar() == 0
    assert conn.execute(sa.text("select count(*) from occupancy_grid")).scalar() == 0


def test_every_migration_above_the_floor_round_trips_with_rows_present():
    dsn = get_settings().postgres.sync_dsn
    assert get_settings().postgres.db.endswith("_test"), "this walk runs DDL; the test database only"
    engine = sa.create_engine(dsn)
    head = _current()
    assert head and head != FLOOR, "nothing above the floor to round-trip"
    with engine.begin() as c:
        seeded = _seed(c)
    engine.dispose()
    try:

        _alembic("downgrade", FLOOR)
        assert _current() == FLOOR
        engine = sa.create_engine(dsn)
        with engine.begin() as c:
            _assert_downgraded(c, seeded)
        engine.dispose()

        _alembic("upgrade", "head")
        assert _current() == head
        engine = sa.create_engine(dsn)
        with engine.begin() as c:
            _assert_reupgraded(c, seeded)
    finally:
        # never leave the test database below head, whatever failed
        _alembic("upgrade", "head")
        engine = sa.create_engine(dsn)
        with engine.begin() as c:
            c.execute(sa.text("delete from settlement_lot where lot_id = :l"), {"l": seeded["lot_id"]})
            c.execute(sa.text("delete from session where session_id in (:a, :b)"),
                      {"a": seeded["real_sid"], "b": seeded["synth_sid"]})
        engine.dispose()
