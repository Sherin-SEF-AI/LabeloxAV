"""An uploaded crop could not search the object plane.

`POST /api/search/similar` takes a frame id, an object id or an image. The first two say which plane they
mean and were routed accordingly; an image says nothing, so it silently meant frames. That leaves the
exemplar query, "here is a picture of the thing, find me more of it", unreachable, even though the pieces to
answer it were all present: the crops are indexed in DINOv3 and `find_similar_objects` already reranks them.

The routing is what these tests pin, by recording which finder was reached. Actually encoding an image needs
a GPU and a model download, and neither is the thing that was broken.
"""

import base64

import numpy as np
import pytest


def _png_bytes() -> str:
    import cv2

    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture()
def spy(monkeypatch):
    """Record which plane a request reached, without touching a GPU or the database."""
    calls = {}

    async def fake_objects(_db, query_vec, **kw):
        calls["plane"] = "object"
        calls["kw"] = kw
        calls["dim"] = len(query_vec)
        return []

    async def fake_frames(_db, query_vec, **kw):
        calls["plane"] = "frame"
        calls["kw"] = kw
        calls["space"] = kw.get("space")
        return []

    import services.intelligence.search.similar as sim
    monkeypatch.setattr(sim, "find_similar_objects", fake_objects)
    monkeypatch.setattr(sim, "find_similar_frames", fake_frames)

    # The two encoders are given their real, different widths so a query vector's length identifies which
    # one produced it. Returning the same width from both would let this suite pass a query built in the
    # wrong space, which is the exact mistake being guarded against.
    class _Dino:
        @staticmethod
        def encode_image(_img):
            return np.zeros(768, dtype=np.float32)

    class _Siglip:
        @staticmethod
        def encode_image(_img):
            return np.zeros(1152, dtype=np.float32)

    import services.intelligence.embed as embed
    monkeypatch.setattr(embed, "dinov3", _Dino)
    monkeypatch.setattr(embed, "siglip2", _Siglip)
    return calls


@pytest.mark.asyncio
async def test_an_uploaded_crop_can_search_the_object_plane(spy):
    """The query that was unreachable."""
    from services.api.routers.search import SimilarIn, search_similar

    out = await search_similar(SimilarIn(image_b64=_png_bytes(), target="object", k=5), None)
    assert spy["plane"] == "object"
    assert out["kind"] == "object" and out["mode"] == "visual"


@pytest.mark.asyncio
async def test_an_uploaded_crop_still_searches_frames_by_default(spy):
    """Existing callers pass no target and must be completely unaffected."""
    from services.api.routers.search import SimilarIn, search_similar

    out = await search_similar(SimilarIn(image_b64=_png_bytes(), k=5), None)
    assert spy["plane"] == "frame"
    assert out["kind"] == "frame"


@pytest.mark.asyncio
async def test_the_object_plane_is_searched_in_the_visual_space(spy):
    """DINOv3, not SigLIP2, because that is the space the crops are indexed in.

    Asking for semantic mode against objects would compare a picture with a text-aligned space for no gain,
    so target wins over mode here rather than producing a query in a space nothing was indexed in.
    """
    from services.api.routers.search import SimilarIn, search_similar

    await search_similar(SimilarIn(image_b64=_png_bytes(), target="object", mode="semantic", k=5), None)
    assert spy["plane"] == "object"
    assert spy["dim"] == 768, "a 1152-wide SigLIP2 vector cannot query the DINOv3 crop index"


@pytest.mark.asyncio
async def test_the_search_filters_are_carried_through(spy):
    """A scoped exemplar search is the useful one: more like this, in this session."""
    from services.api.routers.search import SimilarIn, search_similar

    await search_similar(SimilarIn(image_b64=_png_bytes(), target="object", k=7, min_sim=0.4,
                                   diversity=False), None)
    assert spy["kw"]["k"] == 7
    assert spy["kw"]["min_sim"] == 0.4
    assert spy["kw"]["diversity"] is False


@pytest.mark.asyncio
async def test_a_request_naming_no_source_is_still_refused():
    from fastapi import HTTPException

    from services.api.routers.search import SimilarIn, search_similar

    with pytest.raises(HTTPException):
        await search_similar(SimilarIn(target="object"), None)
