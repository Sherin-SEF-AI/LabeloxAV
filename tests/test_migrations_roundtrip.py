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
    return {"lot_id": lot_id, "real_sid": real_sid, "synth_sid": synth_sid}


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


def _assert_reupgraded(conn, seeded: dict) -> None:
    row = conn.execute(sa.text(
        "select rule, cap_n, llr, sprt, increments from settlement_lot where lot_id = :l"),
        {"l": seeded["lot_id"]}).one()
    assert row.rule == "wilson" and row.cap_n == 0 and row.llr is None, \
        "a lot that lived through the downgrade is a fixed-rule lot: the stricter reading"
    assert row.sprt == {} and row.increments == []
    assert conn.execute(sa.text("select origin from session where session_id = :s"),
                        {"s": seeded["real_sid"]}).scalar() == "real"


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
