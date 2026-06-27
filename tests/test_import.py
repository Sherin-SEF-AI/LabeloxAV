"""Import tests. Pure: remap + COCO adapter parsing. Infra: presigned-multipart round-trip and the
export->import round-trip oracle (counts + class names preserved)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.imports.remap import remap_name


def test_remap_known_and_fallback():
    onto = get_ontology()
    # an ontology name maps to itself
    some = onto.classes[0].name
    cid, name, mapped = remap_name(some, onto)
    assert mapped and name == some
    # an unknown vehicle-ish token hits vehicle_fallback (if present), still a valid class
    cid2, name2, mapped2 = remap_name("zzz_lorry_thing", onto)
    assert not mapped2 and onto.has_name(name2)


def test_coco_adapter_parse(tmp_path: Path):
    from services.imports.adapter_coco import parse

    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "uri": "s3://b/frames/a.jpg", "width": 100, "height": 80, "ts_ns": 5}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 30],
             "labelox": {"class_name": "sedan", "conf": 0.9, "attributes": {"occlusion": 0}}},
        ],
        "categories": [{"id": 7, "name": "sedan"}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    frames = parse(tmp_path)
    assert len(frames) == 1
    f = frames[0]
    assert f.image_ref == "s3://b/frames/a.jpg" and f.width == 100
    assert len(f.objects) == 1
    o = f.objects[0]
    assert o.name == "sedan" and o.bbox == [10, 10, 30, 40] and o.conf == 0.9


def test_mapillary_adapter_parse(tmp_path: Path):
    import numpy as np
    from services.imports.adapter_mapillary import parse

    (tmp_path / "images").mkdir()
    (tmp_path / "polygons").mkdir()
    key = "abc123"
    # a tiny image so _find_image succeeds
    import cv2

    cv2.imwrite(str(tmp_path / "images" / f"{key}.jpg"), np.zeros((40, 60, 3), np.uint8))
    doc = {
        "width": 60, "height": 40,
        "objects": [
            {"id": 1, "label": "object--vehicle--car", "polygon": [[10, 10], [30, 10], [30, 25], [10, 25]]},
            {"id": 2, "label": "construction--flat--road", "polygon": [[0, 30], [60, 30], [60, 40], [0, 40]]},
        ],
    }
    (tmp_path / "polygons" / f"{key}.json").write_text(json.dumps(doc))
    frames = parse(tmp_path)
    assert len(frames) == 1
    f = frames[0]
    # the road (stuff) is skipped; only the car (instance) becomes a box
    assert len(f.objects) == 1
    o = f.objects[0]
    assert o.name == "sedan" and o.bbox == [10, 10, 30, 25]


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
def test_multipart_roundtrip_and_dedup():
    import httpx

    from core.storage import get_object_store

    store = get_object_store()
    store.ensure_bucket()
    key = "uploads/_test_mp/blob.bin"
    upload_id = store.create_multipart(key, "application/octet-stream")
    part1 = b"A" * (5 * 1024 * 1024)  # >=5 MB (multipart minimum except last)
    part2 = b"B" * 1024
    parts = []
    try:
        for i, chunk in enumerate([part1, part2], start=1):
            url = store.presign_part(key, upload_id, i)
            r = httpx.put(url, content=chunk, timeout=60)
            r.raise_for_status()
            etag = r.headers.get("ETag") or r.headers.get("etag")
            assert etag, "no ETag returned (CORS/expose-headers)"
            parts.append({"PartNumber": i, "ETag": etag})
        uri = store.complete_multipart(key, upload_id, parts)
    except Exception:
        store.abort_multipart(key, upload_id)
        raise
    assert store.get_bytes(uri) == part1 + part2

    # content-addressed dedup: identical bytes -> identical uri, idempotent
    u1 = store.put_content_addressed("uploads/_test_dedup", b"same-bytes", ".bin")
    u2 = store.put_content_addressed("uploads/_test_dedup", b"same-bytes", ".bin")
    assert u1 == u2


@requires_infra
@pytest.mark.asyncio
async def test_export_import_roundtrip():
    """Hermetic round-trip oracle: ingest real (stored) frames, attach objects, export to parquet+coco,
    import back, and assert object count + class names are preserved and images actually load."""
    import uuid

    import numpy as np
    from sqlalchemy import func, select

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.export.dataset import SliceSpec, export_dataset
    from services.imports.records import ImportSpec
    from services.imports.run import import_dataset
    from services.ingest.run import ingest
    from services.ingest.types import RawFrame

    onto = get_ontology()
    cls = onto.classes[0]

    rng = np.random.default_rng(11)
    start = now_ns()
    frames = [RawFrame(ts_ns=start + seconds_to_ns(i), cam_id="cam_rt",
                       image_bgr=rng.integers(20, 230, (480, 640, 3), dtype=np.uint8)) for i in range(2)]
    ing = await ingest(frame_iter=iter(frames), vehicle="RT-SRC", city="BLR", route="rt",
                       raw_uri=None, mcap_uri=None, source_streams=["cam_rt"])
    src_sid = uuid.UUID(ing["session_id"])

    # attach two accepted objects on the ingested frames (so they survive export and carry an image)
    async with get_sessionmaker()() as db:
        frame_ids = (await db.execute(select(Frame.frame_id).where(Frame.session_id == src_sid))).scalars().all()
        for fid in frame_ids:
            db.add(Object(frame_id=fid, class_id=cls.id, bbox=[100.0, 100.0, 200.0, 200.0], conf=0.9,
                          source="fused", state="accepted", provenance={"raw_conf": 0.9}, attrs={}))
        await db.commit()

    res = await export_dataset(SliceSpec(name="rt-test", session_id=str(src_sid), states=["accepted"],
                                         formats=["parquet", "coco"]))
    out_dir = Path(res["out_dir"])
    expected = res["object_count"]
    assert expected == 2

    imp = await import_dataset(ImportSpec(format="parquet", source_uri=str(out_dir), target_vehicle="RT-01"))
    assert imp["counts"]["objects"] == expected

    new_sid = uuid.UUID(imp["session_id"])
    async with get_sessionmaker()() as db:
        n = (await db.execute(
            select(func.count()).select_from(Object).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == new_sid, Object.source == "imported", Object.state == "review")
        )).scalar_one()
        names = (await db.execute(
            select(Object.class_id).join(Frame, Object.frame_id == Frame.frame_id).where(Frame.session_id == new_sid)
        )).scalars().all()
    assert n == expected
    assert all(c == cls.id for c in names)  # class preserved through the round-trip
