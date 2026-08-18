"""A contact sheet has to be one request that decodes each frame once.

`GET /api/objects/{id}/crop` fetches the whole frame from the object store, decodes the entire JPEG, cuts
one box out of it and re-encodes, synchronously inside an async handler. That is fine for a filmstrip
thumbnail and unusable for a grid: 200 tiles is 200 whole-frame fetches and decodes, and objects cluster on
frames, so the same JPEG is decoded dozens of times over.

Measured on the live corpus: 60 co-located crops take 1.01 s one at a time and 0.04 s as a sheet, decoding 2
frames instead of 60. Even where the objects share no frames, which is the usual shape of a ranked batch,
collapsing 60 requests into one is 2.5x.

These tests cover the parts that would silently corrupt a grid rather than merely slow it: the order a
reviewer sees, what happens to a tile whose image is gone, and that a crop is not stretched.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _frame_with_objects(db, onto, *, n_objects: int, session_id=None, boxes=None):
    """One frame carrying n objects, so co-location can be exercised."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    cid = next(c.id for c in onto.classes if c.name == "rider")
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts = now_ns()
    sid = session_id or uuid.uuid4()
    if session_id is None:
        db.add(DbSession(session_id=sid, vehicle_id="SHEET-1", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        await db.flush()
    fid = uuid.uuid4()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    oids = []
    for i in range(n_objects):
        oid = uuid.uuid4()
        bb = (boxes[i] if boxes else [10.0 + i, 10.0, 110.0 + i, 210.0])
        db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=list(bb), conf=0.5,
                      source="fused", state="review", attrs={}, provenance={}, version=1))
        oids.append(str(oid))
    await db.flush()
    return sid, str(fid), oids


@pytest.mark.asyncio
async def test_tiles_come_back_in_the_order_they_were_asked_for():
    """The reviewer's order is the ranking, and the database's order is not it."""
    from db.session import get_sessionmaker
    from services.api.routers.objects import CropSheetIn, object_crop_sheet
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        _sid, _fid, oids = await _frame_with_objects(db, get_ontology(), n_objects=6)
        asked = list(reversed(oids))
        out = await object_crop_sheet(CropSheetIn(object_ids=asked, cell=64), db)

        assert [p["object_id"] for p in out["placements"]] == asked
        await db.rollback()


@pytest.mark.asyncio
async def test_a_tile_whose_image_is_missing_keeps_its_place():
    """A dropped tile would shift every later one, so the grid would silently mislabel every verdict.

    These fixtures point at object-store keys that do not exist, which is exactly the case: the crop cannot
    be produced, and the cell must still be emitted so index arithmetic on the client stays true.
    """
    from db.session import get_sessionmaker
    from services.api.routers.objects import CropSheetIn, object_crop_sheet
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        _sid, _fid, oids = await _frame_with_objects(db, get_ontology(), n_objects=4)
        out = await object_crop_sheet(CropSheetIn(object_ids=oids, cell=64), db)

        assert out["count"] == 4
        assert [p["object_id"] for p in out["placements"]] == oids
        assert all(p["ok"] is False for p in out["placements"]), "the fixture images do not exist"
        # Positions still tile the grid in order.
        assert [(p["row"], p["col"]) for p in out["placements"]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        await db.rollback()


@pytest.mark.asyncio
async def test_each_frame_is_decoded_once_however_many_crops_it_holds():
    """The whole reason the endpoint exists. 60 crops on 2 frames must be 2 decodes, not 60."""
    from db.session import get_sessionmaker
    from services.api.routers.objects import CropSheetIn, object_crop_sheet
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        sid, _f1, a = await _frame_with_objects(db, onto, n_objects=10)
        _sid, _f2, b = await _frame_with_objects(db, onto, n_objects=10, session_id=sid)
        out = await object_crop_sheet(CropSheetIn(object_ids=a + b, cell=64), db)

        assert out["crops"] == 20
        assert out["frames_decoded"] == 2, "a frame must be fetched and decoded exactly once"
        await db.rollback()


@pytest.mark.asyncio
async def test_the_sheet_is_bounded_so_one_request_cannot_ask_for_the_corpus():
    from db.session import get_sessionmaker
    from services.api.routers.objects import MAX_SHEET_CROPS, CropSheetIn, object_crop_sheet
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        _sid, _fid, oids = await _frame_with_objects(db, get_ontology(), n_objects=3)
        padded = oids + [str(uuid.uuid4()) for _ in range(MAX_SHEET_CROPS + 50)]
        out = await object_crop_sheet(CropSheetIn(object_ids=padded, cell=64), db)
        assert out["count"] <= MAX_SHEET_CROPS
        await db.rollback()


@pytest.mark.asyncio
async def test_an_empty_request_is_an_empty_sheet_not_an_error():
    """A grid whose filter matched nothing asks for nothing, and that is a normal answer."""
    from db.session import get_sessionmaker
    from services.api.routers.objects import CropSheetIn, object_crop_sheet

    async with get_sessionmaker()() as db:
        out = await object_crop_sheet(CropSheetIn(object_ids=[], cell=64), db)
        assert out["count"] == 0 and out["sheet"] is None and out["placements"] == []


@pytest.mark.asyncio
async def test_the_grid_is_near_square_by_default():
    from db.session import get_sessionmaker
    from services.api.routers.objects import CropSheetIn, object_crop_sheet
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        _sid, _fid, oids = await _frame_with_objects(db, get_ontology(), n_objects=9)
        out = await object_crop_sheet(CropSheetIn(object_ids=oids, cell=64), db)
        assert out["cols"] == 3 and out["rows"] == 3
        await db.rollback()
