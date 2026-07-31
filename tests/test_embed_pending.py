"""A row is not a vector.

The daemon and the backfill both asked "does an embedding row exist" and treated the answer as "is this
object embedded". `object_embedding.siglip_vec` arrived later, in migration 0073, as a nullable column, so
every crop embedded before it has a row, a DINOv3 vector, and a NULL SigLIP2 one. Under a row-existence test
those crops are complete forever, while `core/embeddings.py:object_neighbors_by_text` filters on
`siglip_vec IS NOT NULL` and cannot see them.

On the live corpus that was total and silent: all 570,305 object embeddings had a NULL `siglip_vec`, so
object text search returned zero results for every query while every counter reported full coverage.

These tests construct that exact shape, a half-embedded object, and pin that it is now counted as pending.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns


async def _object_with_embedding(db, onto, *, dino: bool, siglip: bool):
    """One object, optionally carrying each vector. Returns its id."""
    from db.models import Frame, Object, ObjectEmbedding, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    cid = next(c.id for c in onto.classes if c.name == "bus")
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts, sid, fid, oid = now_ns(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="EMB-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri="s3://x/e.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[10.0, 10.0, 90.0, 90.0],
                  conf=0.8, source="fused", state="review", attrs={}, provenance={}, version=1))
    await db.flush()
    if dino or siglip:
        db.add(ObjectEmbedding(
            object_id=oid,
            dino_vec=([0.1] * 768) if dino else None,
            siglip_vec=([0.2] * 1152) if siglip else None,
            model_versions={},
        ))
        await db.flush()
    return oid


async def _is_pending(db, oid) -> bool:
    from sqlalchemy import select

    from db.models import Object
    from services.intelligence.embed.pending import object_needs_embedding

    rows = (await db.execute(
        select(Object.object_id).where(Object.object_id == oid, object_needs_embedding()))).scalars().all()
    return len(rows) == 1


@pytest.mark.asyncio
async def test_an_object_with_no_embedding_row_is_pending():
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oid = await _object_with_embedding(db, get_ontology(), dino=False, siglip=False)
        assert await _is_pending(db, oid) is True
        await db.rollback()


@pytest.mark.asyncio
async def test_an_object_with_both_vectors_is_not_pending():
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oid = await _object_with_embedding(db, get_ontology(), dino=True, siglip=True)
        assert await _is_pending(db, oid) is False
        await db.rollback()


@pytest.mark.asyncio
async def test_an_object_missing_only_siglip_is_pending():
    """The defect. This is every crop in the corpus, and it counted as done."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oid = await _object_with_embedding(db, get_ontology(), dino=True, siglip=False)
        assert await _is_pending(db, oid) is True, \
            "a crop with a DINOv3 vector and no SigLIP2 one is unreachable by text search and must be re-embedded"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_old_predicate_called_it_complete():
    """What the daemon and the backfill both did, kept executable so the regression stays demonstrable."""
    from sqlalchemy import exists, select

    from db.models import Object, ObjectEmbedding
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        oid = await _object_with_embedding(db, get_ontology(), dino=True, siglip=False)

        old_pending = (await db.execute(select(Object.object_id).where(
            Object.object_id == oid,
            ~exists().where(ObjectEmbedding.object_id == Object.object_id)))).scalars().all()
        assert old_pending == [], "precondition: the row-existence test saw nothing to do"
        assert await _is_pending(db, oid) is True, "the vector-completeness test does"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_half_embedded_population_is_countable_on_its_own():
    """A number that used to read zero needs its own counter, or the fix is unverifiable."""
    from sqlalchemy import func, select

    from db.models import Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.intelligence.embed.pending import object_missing_siglip

    async with get_sessionmaker()() as db:
        before = (await db.execute(
            select(func.count()).select_from(Object).where(object_missing_siglip()))).scalar_one()
        await _object_with_embedding(db, get_ontology(), dino=True, siglip=False)
        after = (await db.execute(
            select(func.count()).select_from(Object).where(object_missing_siglip()))).scalar_one()
        assert after == before + 1
        await db.rollback()
