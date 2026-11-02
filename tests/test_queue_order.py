"""The review queue's order: one formula, ranked across the whole pool, and measured on gold.

The queue used to over-fetch an unordered three pages and rank those, which served the top of whatever
Postgres read first. The rank now runs in SQL; these tests hold the SQL and Python forms of the formula
together, check the queue returns the true top of its pool, and check the gold measurement behind the choice.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _classes():
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rare = next(c for c in onto.classes if c.india and onto.is_thing(c.id))
    common = next(c for c in onto.classes if not c.india and c.l1 != "fallback" and onto.is_thing(c.id))
    return onto, common.id, rare.id


# (class, conf, provenance) covering every term: rarity, mask/box disagreement, class conflict, plain.
def _variety(common: int, rare: int) -> list[tuple[int, float, dict]]:
    conflict = {"proposals": [{"verdict": "agree"}, {"verdict": "overruled"}]}
    return [
        (common, 0.95, {}), (common, 0.40, {}), (rare, 0.70, {}), (rare, 0.30, {"mask_box_disagree": True}),
        (common, 0.55, conflict), (rare, 0.62, {**conflict, "mask_box_disagree": True}),
        (common, 0.20, {"proposals": "not-a-list"}), (common, 0.50, {"proposals": [{"verdict": "overruled"}]}),
    ]


async def _seed(rows: list[tuple[int, float, dict]], *, state: str = "review", source: str = "fused") -> tuple[str, list[str]]:
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    sid, ts = uuid.uuid4(), now_ns()
    oids = []
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="QORD-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version=get_ontology().version))
        await db.flush()
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/0.jpg",
                     width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        for i, (cid, conf, prov) in enumerate(rows):
            oid = uuid.uuid4()
            oids.append(str(oid))
            x = 10.0 + 60 * i
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[x, 10.0, x + 50, 60.0], conf=conf,
                          source=source, state=state, attrs={}, provenance=prov, version=1))
        await db.commit()
    return str(sid), oids


def test_the_sql_rank_is_the_python_rank():
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.queue_order import priority_of, priority_sql, rare_class_ids

    onto, common, rare = _classes()
    rows = _variety(common, rare)
    _, oids = _run(_seed(rows))
    rare_ids = rare_class_ids(onto)

    async def _read():
        async with get_sessionmaker()() as db:
            return (await db.execute(select(Object.object_id, priority_sql(rare_ids))
                                     .where(Object.object_id.in_([uuid.UUID(o) for o in oids])))).all()

    sql = {str(i): float(p) for i, p in _run(_read())}
    for oid, (cid, conf, prov) in zip(oids, rows, strict=True):
        assert sql[oid] == pytest.approx(priority_of(conf, cid, prov, set(rare_ids)), abs=1e-4), (cid, conf, prov)


def test_the_queue_is_the_top_of_its_whole_pool():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app
    from services.queue_order import priority_of, rare_class_ids

    onto, common, rare = _classes()
    # Enough rows that a page of 5 over-fetched 3x and ranked would miss most of the true top.
    rows = [(common, 0.99 - 0.004 * i, {}) for i in range(40)] + _variety(common, rare)
    sid, _ = _run(_seed(rows))
    rare_ids = set(rare_class_ids(onto))
    expected = sorted((priority_of(c, cid, p, rare_ids) for cid, c, p in rows), reverse=True)[:5]
    with TestClient(app) as c:
        r = c.get("/api/triage", params={"session_id": sid, "limit": 5}, headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        assert [row["priority"] for row in r.json()] == pytest.approx(expected, abs=1e-4)


def _seed_gold(wrong_low_conf: bool) -> str:
    """A gold set over one frame, with machine labels on top of each gold object: half the class of the gold
    label (right), half another class (wrong). When `wrong_low_conf`, the wrong ones are the unsure ones."""
    from db.models import Frame, GoldSet, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    onto, common, rare = _classes()
    other = next(c.id for c in onto.classes if not c.india and onto.is_thing(c.id) and c.id != common)

    async def _go():
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        gold_ids = []
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="QGOLD", start_ts_ns=ts, end_ts_ns=ts + 1, city="BLR",
                             sensors={}, ontology_version=onto.version))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/g.jpg",
                         width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            for i in range(20):
                x = 10.0 + 80 * i
                g = Object(object_id=uuid.uuid4(), frame_id=fid, class_id=common, bbox=[x, 10.0, x + 60, 70.0],
                           conf=1.0, source="human", state="accepted", attrs={}, provenance={}, version=1)
                db.add(g)
                gold_ids.append(str(g.object_id))
                wrong = i % 2 == 0
                conf = (0.3 if wrong else 0.9) if wrong_low_conf else 0.6
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=other if wrong else common,
                              bbox=[x + 2, 12.0, x + 62, 72.0], conf=conf, source="fused", state="review",
                              attrs={}, provenance={}, version=1))
            gid = f"gold-test-{uuid.uuid4().hex[:12]}"
            db.add(GoldSet(gold_id=gid, name="queue-order-test", object_ids=gold_ids, n_objects=20, n_frames=1,
                           ontology_version=onto.version))
            await db.commit()
        return gid

    return _run(_go())


def test_the_gold_measurement_credits_an_ordering_that_puts_wrong_labels_first():
    from db.session import get_sessionmaker
    from services.queue_order import evaluate_orderings

    gid = _seed_gold(wrong_low_conf=True)

    async def _eval():
        async with get_sessionmaker()() as db:
            return await evaluate_orderings(db, gid, seeds=50)

    out = _run(_eval())
    assert out["pool"] == 20 and out["wrong"] == 10 and out["wrong_rate"] == 0.5
    q, rnd = out["orderings"]["queue_formula"], out["orderings"]["random"]
    # All ten wrong labels are the unsure ones, so the queue meets every one of them in its first half.
    assert q["found"][2] == 1.0 and q["ap"] == 1.0
    assert rnd["found"][2] == pytest.approx(0.5, abs=0.1)
    assert out["best"] in ("queue_formula", "lowest_confidence")


def test_the_ordering_endpoint_measures_the_newest_gold_set_or_says_there_is_none():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    gid = _seed_gold(wrong_low_conf=False)
    with TestClient(app) as c:
        r = c.get("/api/triage/ordering", params={"gold_id": gid}, headers=auth_headers("annotator"))
        assert r.status_code == 200, r.text
        assert r.json()["gold_id"] == gid and set(r.json()["orderings"]) >= {"queue_formula", "random"}
        assert c.get("/api/triage/ordering", params={"gold_id": "gold-nope"},
                     headers=auth_headers("annotator")).status_code == 404
