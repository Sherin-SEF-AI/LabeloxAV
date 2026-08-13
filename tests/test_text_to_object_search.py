"""Text-to-object retrieval: find a crop by describing it.

Two defects:

1. `ObjectEmbedding` carried only a DINOv3 vector. DINOv3 has no text tower, so a phrase could retrieve whole
   frames but never the objects inside them, which is the query a reviewer actually wants ("cattle at night"
   should return cattle crops, not frames to scan by eye).
2. `search_objects_by_text` did exist, but it loaded the entire legacy CLIP `Embedding` table into memory and
   cosined it in Python. That table is no longer populated by the pipeline, so on a live corpus the search
   quietly returned nothing, and even populated it was a full scan rather than an index lookup.

The fix stores a SigLIP2 vector per crop (a joint image-text space) with an HNSW index, and routes the search
through it. These tests use precomputed unit vectors rather than running the encoder, so they assert the
retrieval and filtering logic without needing a GPU."""
from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db

_DIM = 1152


def _infra_up() -> bool:
    from core.config import get_settings
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _unit(index: int) -> list[float]:
    """A one-hot unit vector: orthogonal to every other, so cosine ranking is exactly predictable."""
    v = [0.0] * _DIM
    v[index] = 1.0
    return v


async def _seed(db, vectors: list[list[float] | None]):
    """One session/frame with an object per vector. A None vector means the crop is not embedded yet."""
    from db.models import Frame, Object, ObjectEmbedding
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    db.add(DbSession(session_id=sid, vehicle_id="TXT-01", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/y.jpg",
                 width=640, height=480, quality=1.0))
    await db.flush()

    oids = []
    for i, vec in enumerate(vectors):
        oid = uuid.uuid4()
        oids.append(oid)
        db.add(Object(object_id=oid, frame_id=fid, class_id=onto.by_name("sedan").id,
                      bbox=[i * 10.0, 0.0, i * 10.0 + 50, 50.0], conf=0.9, attrs={},
                      source="human", state="accepted"))
        await db.flush()
        db.add(ObjectEmbedding(object_id=oid, dino_vec=[0.0] * 768, siglip_vec=vec, model_versions={}))
    await db.commit()
    return sid, oids


@requires_infra
async def test_nearest_crop_in_the_shared_space_ranks_first(monkeypatch):
    from core import embeddings as emb
    from db.session import get_sessionmaker

    # Stand in for the encoder: the query encodes to the same direction as the second crop.
    monkeypatch.setattr(emb, "prompt_vector", lambda text, s: _unit(5))

    async with get_sessionmaker()() as db:
        sid, oids = await _seed(db, [_unit(1), _unit(5), _unit(9)])
        hits = await emb.object_neighbors_by_text(db, "anything", k=5, session_id=sid)

    assert hits, "the query must return the embedded crops"
    assert hits[0][0] == str(oids[1])
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)   # identical direction, cosine similarity 1


@requires_infra
async def test_unembedded_crops_are_skipped_not_ranked_poorly(monkeypatch):
    # A null vector is an absence of evidence. Ranking it as a distant match would list an object that was
    # never actually considered, as though the search had looked at it and judged it.
    from core import embeddings as emb
    from db.session import get_sessionmaker

    monkeypatch.setattr(emb, "prompt_vector", lambda text, s: _unit(3))

    async with get_sessionmaker()() as db:
        sid, oids = await _seed(db, [_unit(3), None, None])
        hits = await emb.object_neighbors_by_text(db, "anything", k=10, session_id=sid)

    assert [h[0] for h in hits] == [str(oids[0])]


@requires_infra
async def test_session_filter_scopes_the_search(monkeypatch):
    from core import embeddings as emb
    from db.session import get_sessionmaker

    monkeypatch.setattr(emb, "prompt_vector", lambda text, s: _unit(7))

    async with get_sessionmaker()() as db:
        sid_a, oids_a = await _seed(db, [_unit(7)])
        sid_b, _ = await _seed(db, [_unit(7)])
        hits = await emb.object_neighbors_by_text(db, "anything", k=10, session_id=sid_a)

    assert [h[0] for h in hits] == [str(oids_a[0])], "another session's crop must not leak into the result"


@requires_infra
async def test_class_filter_scopes_the_search(monkeypatch):
    from core import embeddings as emb
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    monkeypatch.setattr(emb, "prompt_vector", lambda text, s: _unit(2))
    onto = get_ontology()

    async with get_sessionmaker()() as db:
        sid, _ = await _seed(db, [_unit(2)])
        # the seeded objects are sedans; asking for pedestrians must return nothing rather than the sedan
        hits = await emb.object_neighbors_by_text(
            db, "anything", k=10, session_id=sid, class_id=onto.by_name("pedestrian").id)

    assert hits == []


@requires_infra
async def test_min_sim_drops_weak_matches(monkeypatch):
    from core import embeddings as emb
    from db.session import get_sessionmaker

    monkeypatch.setattr(emb, "prompt_vector", lambda text, s: _unit(4))

    async with get_sessionmaker()() as db:
        sid, oids = await _seed(db, [_unit(4), _unit(11)])   # one identical, one orthogonal
        hits = await emb.object_neighbors_by_text(db, "anything", k=10, session_id=sid, min_sim=0.5)

    assert [h[0] for h in hits] == [str(oids[0])]


def test_query_is_wrapped_in_the_caption_template():
    # SigLIP was trained on captions, so a bare noun phrase sits off-distribution; the crop classifier already
    # uses this template and retrieval must match it or the two disagree about what a class looks like.
    from core.embeddings import prompt_vector

    seen: list[str] = []

    class _Enc:
        @staticmethod
        def encode_text(t: str):
            seen.append(t)
            return [0.0] * _DIM

    prompt_vector("cattle at night", _Enc())
    prompt_vector("a photo of a bus", _Enc())
    assert seen == ["a photo of cattle at night", "a photo of a bus"]   # not double-wrapped
