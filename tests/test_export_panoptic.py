"""COCO panoptic export: the layer that could be created and corrected but never delivered.

`FrameSegmentation` holds semantic and panoptic rasters with a `human` source, and no write path out. That
is the same failure `adapter_scene.py` was written to fix for masks, lanes and drivable surfaces: a
visualisation rather than a deliverable.

The tests are shaped around the two ways a panoptic export is wrong while still loading: emitting a class
map instead of an id map, and packing segment ids into pixels in the wrong channel order, which is invisible
until a frame has more than 255 segments.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid

import cv2
import numpy as np
import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


def _npz(store, key: str, arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr)
    return store.put_bytes(key, buf.getvalue(), "application/octet-stream")


def _decode_ids(png_path) -> np.ndarray:
    """The id map, read back the way a COCO panoptic consumer reads it: id = R + G*256 + B*65536."""
    im = cv2.imread(str(png_path))          # cv2 returns BGR
    return (im[:, :, 2].astype(np.int64)
            + im[:, :, 1].astype(np.int64) * 256
            + im[:, :, 0].astype(np.int64) * 65536)


async def _seed(kind: str, labels: np.ndarray, instances: np.ndarray | None):
    """One frame with a stored raster, and the ids needed to export it."""
    from core.storage import get_object_store
    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, FrameSegmentation
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()

    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="CP-01", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                     img_uri="s3://x/f.jpg", width=labels.shape[1], height=labels.shape[0],
                     quality=0.9, scene={}))
        await db.flush()
        db.add(FrameSegmentation(
            frame_id=fid, kind=kind,
            labels_uri=_npz(store, f"seg/{fid}/labels.npz", labels),
            instance_uri=(_npz(store, f"seg/{fid}/inst.npz", instances) if instances is not None else None),
            coverage={}, segments={}, source="human", ontology_version=onto.version))
        await db.commit()
    return str(fid)


async def _export(frame_ids, tmp_path):
    from core.storage import get_object_store
    from services.export.adapter_scene import write_panoptic

    out = await write_panoptic(frame_ids, get_object_store(), tmp_path / "p")
    return json.loads((out / "panoptic.json").read_text()), out


def _classes():
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    car = next(c.id for c in onto.classes if c.name == "sedan")       # a thing
    road = next(c.id for c in onto.classes if c.name == "road")       # stuff
    return car, road


# --- the central correctness property ------------------------------------------------------------


@requires_infra
def test_the_png_is_an_id_map_not_a_class_map(tmp_path):
    """The whole point of panoptic. Two cars of the same class must be two segments.

    Emitting the class raster and calling it panoptic produces a file that loads and scores nonsense,
    because every car in the frame is one blob.
    """
    car, road = _classes()

    async def _flow():
        labels = np.full((40, 60), road, dtype=np.int32)
        labels[5:15, 5:15] = car
        labels[5:15, 30:40] = car          # a second car, same class
        instances = np.zeros((40, 60), dtype=np.int32)
        instances[5:15, 5:15] = 1
        instances[5:15, 30:40] = 2

        fid = await _seed("panoptic", labels, instances)
        doc, out = await _export([fid], tmp_path)

        segs = doc["annotations"][0]["segments_info"]
        car_segs = [s for s in segs if s["category_id"] == car]
        assert len(car_segs) == 2, "two instances of one class must be two segments"
        assert len({s["id"] for s in segs}) == len(segs), "segment ids must be unique within a frame"

        ids = _decode_ids(out / "panoptic" / f"{fid}.png")
        assert len({int(v) for v in np.unique(ids)}) == len(segs) + (1 if 0 in ids else 0)

    run_async(_flow())


@requires_infra
def test_a_segment_id_above_255_survives_the_pixel_encoding(tmp_path):
    """The channel-order trap. cv2 writes BGR, so packing the id little-endian across RGB and handing it
    straight to imwrite silently swaps the high and low bytes. Invisible until a frame has 256 segments,
    at which point every id is wrong and nothing errors."""
    car, road = _classes()

    async def _flow():
        # 300 distinct instances, so ids run past the single-byte boundary.
        labels = np.full((20, 320), road, dtype=np.int32)
        instances = np.zeros((20, 320), dtype=np.int32)
        for i in range(300):
            labels[5:15, i] = car
            instances[5:15, i] = i + 1

        fid = await _seed("panoptic", labels, instances)
        doc, out = await _export([fid], tmp_path)

        segs = doc["annotations"][0]["segments_info"]
        assert len(segs) > 255, f"only {len(segs)} segments; this test needs to cross the byte boundary"

        ids = _decode_ids(out / "panoptic" / f"{fid}.png")
        declared = {s["id"] for s in segs}
        found = {int(v) for v in np.unique(ids)} - {0}
        assert found == declared, "the ids in the PNG must be the ids segments_info declares"
        assert max(declared) > 255

    run_async(_flow())


# --- what the categories say ----------------------------------------------------------------------


@requires_infra
def test_isthing_comes_from_the_ontology_not_from_a_guess(tmp_path):
    """The thing/stuff split already exists because the persist chokepoint uses it, and it is the same
    distinction COCO panoptic needs. Two sources for one fact would drift."""
    car, road = _classes()

    async def _flow():
        labels = np.full((20, 20), road, dtype=np.int32)
        labels[2:8, 2:8] = car
        instances = np.zeros((20, 20), dtype=np.int32)
        instances[2:8, 2:8] = 1

        fid = await _seed("panoptic", labels, instances)
        doc, _ = await _export([fid], tmp_path)

        by_id = {c["id"]: c for c in doc["categories"]}
        assert by_id[car]["isthing"] == 1
        assert by_id[road]["isthing"] == 0

    run_async(_flow())


@requires_infra
def test_area_and_bbox_describe_the_segment(tmp_path):
    car, road = _classes()

    async def _flow():
        labels = np.full((40, 40), road, dtype=np.int32)
        labels[10:20, 5:15] = car        # 10x10 block at (5, 10)
        instances = np.zeros((40, 40), dtype=np.int32)
        instances[10:20, 5:15] = 1

        fid = await _seed("panoptic", labels, instances)
        doc, _ = await _export([fid], tmp_path)

        seg = next(s for s in doc["annotations"][0]["segments_info"] if s["category_id"] == car)
        assert seg["area"] == 100
        assert seg["bbox"] == [5, 10, 10, 10]     # COCO bbox is [x, y, w, h]

    run_async(_flow())


# --- the honest part ------------------------------------------------------------------------------


@requires_infra
def test_a_semantic_raster_is_exported_but_marked_as_having_merged_instances(tmp_path):
    """A semantic raster has no instance channel, so a thing class becomes one blob covering every
    instance. Refusing those frames would drop most of the rasters this corpus holds; converting them
    silently would corrupt anybody's PQ score."""
    car, road = _classes()

    async def _flow():
        labels = np.full((20, 40), road, dtype=np.int32)
        labels[5:15, 2:8] = car
        labels[5:15, 20:26] = car        # visually two cars, one semantic region

        fid = await _seed("semantic", labels, None)
        doc, _ = await _export([fid], tmp_path)

        ann = doc["annotations"][0]
        assert ann["kind"] == "semantic"
        car_segs = [s for s in ann["segments_info"] if s["category_id"] == car]
        assert len(car_segs) == 1, "without an instance channel a thing class is one segment"
        assert "sedan" in (ann["instances_merged_from_semantic"] or [])
        assert doc["frames_with_merged_instances"] == 1
        # and stated at the top level, not only per annotation
        assert "merges every instance" in doc["note"]

    run_async(_flow())


@requires_infra
def test_a_stuff_class_from_a_semantic_raster_is_not_flagged(tmp_path):
    """For stuff, one segment per class is exactly right and loses nothing. Flagging it would train
    consumers to ignore the flag."""
    _car, road = _classes()

    async def _flow():
        fid = await _seed("semantic", np.full((10, 10), road, dtype=np.int32), None)
        doc, _ = await _export([fid], tmp_path)
        assert doc["annotations"][0]["instances_merged_from_semantic"] is None
        assert doc["frames_with_merged_instances"] == 0

    run_async(_flow())


@requires_infra
def test_an_unreadable_raster_costs_its_frame_not_the_export(tmp_path):
    from db.models import FrameSegmentation
    from db.session import get_sessionmaker

    car, road = _classes()

    async def _flow():
        labels = np.full((10, 10), road, dtype=np.int32)
        labels[2:5, 2:5] = car
        good = await _seed("semantic", labels, None)
        bad = await _seed("semantic", labels, None)

        # Point one row at a uri that is not there.
        async with get_sessionmaker()() as db:
            from sqlalchemy import update
            await db.execute(update(FrameSegmentation)
                             .where(FrameSegmentation.frame_id == uuid.UUID(bad))
                             .values(labels_uri="s3://labeloxav/does/not/exist.npz"))
            await db.commit()

        doc, _ = await _export([good, bad], tmp_path)
        assert len(doc["images"]) == 1
        assert doc["unreadable"] == 1, "the shortfall has to be reported, not left to be noticed"

    run_async(_flow())


@requires_infra
def test_a_frame_with_both_kinds_exports_the_panoptic_one(tmp_path):
    """Exporting the semantic duplicate over the panoptic row would lose instances for no reason."""
    from core.storage import get_object_store
    from db.models import FrameSegmentation
    from db.session import get_sessionmaker

    car, road = _classes()

    async def _flow():
        labels = np.full((20, 20), road, dtype=np.int32)
        labels[2:8, 2:8] = car
        instances = np.zeros((20, 20), dtype=np.int32)
        instances[2:8, 2:8] = 1

        fid = await _seed("panoptic", labels, instances)
        store = get_object_store()
        async with get_sessionmaker()() as db:
            db.add(FrameSegmentation(
                frame_id=uuid.UUID(fid), kind="semantic",
                labels_uri=_npz(store, f"seg/{fid}/labels2.npz", labels),
                instance_uri=None, coverage={}, segments={}, source="proposed"))
            await db.commit()

        doc, _ = await _export([fid], tmp_path)
        assert len(doc["annotations"]) == 1
        assert doc["annotations"][0]["kind"] == "panoptic"

    run_async(_flow())


def test_panoptic_is_a_registered_export_target():
    from services.export.dataset import _SCENE_WRITERS, SUPPORTED_EXPORT_FORMATS

    assert "panoptic" in _SCENE_WRITERS
    assert "panoptic" in SUPPORTED_EXPORT_FORMATS


@requires_infra
def test_an_empty_slice_writes_a_valid_empty_document(tmp_path):
    """A consumer should get a loadable file rather than a missing one."""
    async def _flow():
        doc, _ = await _export([], tmp_path)
        assert doc["images"] == [] and doc["categories"] == []
        assert doc["format"] == "coco_panoptic"

    run_async(_flow())


def test_the_export_menu_offers_exactly_what_the_driver_dispatches():
    """The same drift guard the import side got. A format in the menu with no writer is a 400 from a menu
    entry; a writer with no menu entry is a capability nobody can reach, which is what panoptic, lanes,
    drivable and hdmap all were."""
    import re
    from pathlib import Path

    from services.export.dataset import SUPPORTED_EXPORT_FORMATS

    src = Path(__file__).resolve().parent.parent / "web" / "lib" / "menus.ts"
    block = re.search(r"const EXPORT_FORMATS[^=]*=\s*\[(.*?)\];", src.read_text(), re.S)
    assert block, "EXPORT_FORMATS not found; this guard needs updating with the file"
    listed = set(re.findall(r'\["([a-z0-9_]+)"', block.group(1)))

    assert listed - SUPPORTED_EXPORT_FORMATS == set(), (
        f"the menu offers targets the driver cannot write: {sorted(listed - SUPPORTED_EXPORT_FORMATS)}")
    assert SUPPORTED_EXPORT_FORMATS - listed == set(), (
        f"the driver writes targets the menu never offers: {sorted(SUPPORTED_EXPORT_FORMATS - listed)}")
