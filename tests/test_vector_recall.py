"""A filtered vector search returned zero rows and called it "no neighbours".

HNSW visits `ef_search` candidates and stops. The WHERE clause is applied to those candidates, so when none
of them satisfies it the query returns nothing rather than looking further. Measured on the live corpus: a
search filtered to `hoarding`, a class with 26,216 embedded objects, returned zero at every `ef_search` up to
the 1,000 maximum, because the query object sat inside a cluster of 2,354 near-identical crops that filled
the whole window. Nothing in an empty result distinguishes that from a corpus with no matches.

pgvector 0.8's iterative scan keeps going when the filter eats the window, which is what
`core/embeddings.tune_recall` now switches on. Confirmed against the live corpus: the same query returned 5
rows with `hnsw.iterative_scan = relaxed_order` and 0 without it.

What these tests can and cannot do. Starvation needs a graph large enough for HNSW to behave like HNSW; a
test table of a few hundred vectors traverses nearly all of them whatever the window, so the empty result
cannot be reproduced here at any ef_search, and a test claiming otherwise would be theatre. Measured while
writing this: 300 decoys around one query point, index forced, ef_search 10, 40 and 200, iterative scan on
and off, all six combinations returned rows. So what is pinned here is the settings being applied, the
transaction surviving a server that lacks them, and the filtered path returning results at all. The
behaviour itself rests on the corpus measurement above.
"""

from __future__ import annotations

import secrets
import uuid

import numpy as np
import pytest
from sqlalchemy import text

from core.embeddings import HNSW_ITERATIVE_SCAN, object_neighbors, tune_recall
from db.models import Frame, Object, ObjectEmbedding
from db.models import Session as DbSession
from db.session import get_sessionmaker

pytestmark = pytest.mark.db

DINO_DIM = 768
# Comfortably more than the ef_search window, so the decoys alone can crowd out the target class.
DECOYS = 260


def _unit(rng) -> list[float]:
    v = rng.normal(size=DINO_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near(base, rng, scale=0.01) -> list[float]:
    v = np.asarray(base, dtype=np.float32) + rng.normal(scale=scale, size=DINO_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


async def _tight_cluster(db, *, decoy_class: int, wanted_class: int):
    """The shape that breaks a plain HNSW scan: a dense cluster of one class, and one object of another
    class sitting further out. Without iterative scan the cluster fills the window and the wanted object is
    never reached."""
    rng = np.random.default_rng(secrets.randbelow(2**31))
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="VEC-01", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()

    base = _unit(rng)
    qid = uuid.uuid4()
    db.add(Object(object_id=qid, frame_id=fid, class_id=decoy_class, bbox=[1, 1, 9, 9],
                  conf=0.5, source="auto_accept", state="review"))
    await db.flush()
    db.add(ObjectEmbedding(object_id=qid, dino_vec=base, model_versions={}))

    for _ in range(DECOYS):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=decoy_class, bbox=[1, 1, 9, 9],
                      conf=0.5, source="auto_accept", state="review"))
        await db.flush()
        db.add(ObjectEmbedding(object_id=oid, dino_vec=_near(base, rng, 0.005), model_versions={}))

    wanted = uuid.uuid4()
    db.add(Object(object_id=wanted, frame_id=fid, class_id=wanted_class, bbox=[1, 1, 9, 9],
                  conf=0.5, source="auto_accept", state="review"))
    await db.flush()
    db.add(ObjectEmbedding(object_id=wanted, dino_vec=_near(base, rng, 0.35), model_versions={}))
    await db.commit()
    return base, str(wanted)


async def _force_index(db):
    """Make the planner use the HNSW index, which is the only path where this bug exists.

    The live table holds 567,527 vectors so the index is always chosen. A test table holds a few hundred, and
    Postgres sensibly picks a sequential scan, which filters perfectly and would let this test pass against
    the very code it is meant to catch. Forcing the index is what makes it a regression test rather than a
    description of one.
    """
    await db.execute(text("SET LOCAL enable_seqscan = off"))


class TestFilteredSearchKeepsLooking:
    async def test_a_class_hidden_behind_a_tight_cluster_is_still_found(self):
        """The reported shape, at test scale. It does not starve here (see the module docstring), so this
        guards the filtered path rather than reproducing the starvation."""
        async with get_sessionmaker()() as db:
            base, wanted = await _tight_cluster(db, decoy_class=163, wanted_class=122)
            await _force_index(db)
            hits = await object_neighbors(db, base, k=5, class_id=122)
        assert hits, "a filtered search over a tight cluster returned nothing"
        assert wanted in {oid for oid, _ in hits}

    async def test_an_unfiltered_search_is_unaffected(self):
        async with get_sessionmaker()() as db:
            base, _ = await _tight_cluster(db, decoy_class=163, wanted_class=122)
            await _force_index(db)
            hits = await object_neighbors(db, base, k=5)
        assert len(hits) == 5


class TestTheSettings:
    async def test_iterative_scan_is_actually_on_after_tuning(self):
        async with get_sessionmaker()() as db:
            await tune_recall(db)
            got = (await db.execute(text("SHOW hnsw.iterative_scan"))).scalar()
        assert got == HNSW_ITERATIVE_SCAN

    async def test_tuning_is_scoped_to_its_transaction(self):
        """SET LOCAL, so a search cannot change how unrelated queries in the same session behave."""
        async with get_sessionmaker()() as db:
            await tune_recall(db)
            await db.rollback()
            got = (await db.execute(text("SHOW hnsw.iterative_scan"))).scalar()
        assert got != HNSW_ITERATIVE_SCAN or got == "off"

    async def test_a_missing_setting_does_not_poison_the_transaction(self):
        """The first attempt at this issued every SET inside its own try. A failed SET aborts the surrounding
        transaction in Postgres, so on a server without the GUC the next query died with "current transaction
        is aborted" - a worse failure than the one being fixed. Settings are checked for first instead."""
        import core.embeddings as emb

        original = emb._HNSW_GUCS
        emb._HNSW_GUCS = {"hnsw.ef_search", "hnsw.does_not_exist"}
        try:
            async with get_sessionmaker()() as db:
                await tune_recall(db)
                assert (await db.execute(text("SELECT 1"))).scalar() == 1
        finally:
            emb._HNSW_GUCS = original
