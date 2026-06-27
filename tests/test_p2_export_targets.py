"""P2 export targets: OpenLABEL + nuScenes adapters. Unit tests on table/doc structure (no infra),
plus an integration export through dataset.py (DB + MinIO)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns
from services.autolabel.ontology import get_ontology
from services.export.adapter_nuscenes import build_nuscenes
from services.export.adapter_openlabel import write_openlabel
from services.export.records import ExportRecord


def _rec(object_id, frame_id, ts, class_id, class_name, bbox, track_id=None, attrs=None, mask_uri=None):
    return ExportRecord(
        object_id=object_id, frame_id=frame_id, session_id=uuid.UUID(int=1), ts_ns=ts, cam_id="cam_f",
        img_uri=f"s3://x/{frame_id}.jpg", width=640, height=480, vehicle_id="TIGOR-07", city="BLR",
        class_id=class_id, class_name=class_name, bbox=bbox, conf=0.9, state="auto_accept",
        source="auto_accept", mask_uri=mask_uri, track_id=track_id, attrs=attrs or {},
    )


class _FakeStore:
    def get_bytes(self, uri):  # never called when mask_uri is None
        raise AssertionError("should not fetch")


def test_openlabel_structure_and_attributes(tmp_path):
    onto = get_ontology()
    f1, f2 = uuid.uuid4(), uuid.uuid4()
    a = _rec(uuid.uuid4(), f1, 1000, 6, "autorickshaw", [100, 100, 200, 200],
             attrs={"overload": True, "occlusion": 50})
    b = _rec(uuid.uuid4(), f2, 2000, 11, "sedan", [10, 10, 60, 60])
    path = write_openlabel([a, b], onto, _FakeStore(), tmp_path)
    doc = json.loads(path.read_text())["openlabel"]

    assert doc["metadata"]["ontology_version"] == onto.version
    assert len(doc["objects"]) == 2
    assert len(doc["frames"]) == 2
    # frame 0 carries object a with a bbox shape [cx,cy,w,h]
    obj_a = doc["frames"]["0"]["objects"][str(a.object_id)]
    assert obj_a["object_data"]["bbox"][0]["val"] == [150.0, 150.0, 100.0, 100.0]
    # typed attributes ride natively
    block = doc["objects"][str(a.object_id)]["object_data"]
    assert {"name": "overload", "val": True} in block["boolean"]
    assert any(n["name"] == "occlusion" and n["val"] == 50 for n in block["num"])
    assert any(t["name"] == "state" for t in block["text"])


def test_nuscenes_tables_link_and_group_by_track():
    onto = get_ontology()
    f1, f2 = uuid.uuid4(), uuid.uuid4()
    trk = uuid.uuid4()
    a = _rec(uuid.uuid4(), f1, 1_000_000, 6, "autorickshaw", [100, 100, 200, 200], track_id=trk)
    c = _rec(uuid.uuid4(), f1, 1_000_000, 11, "sedan", [300, 300, 360, 360])
    b = _rec(uuid.uuid4(), f2, 2_000_000, 6, "autorickshaw", [120, 100, 220, 200], track_id=trk)
    t = build_nuscenes([a, c, b], onto)

    assert len(t["category"]) == len(onto.classes)  # one nuScenes category per ontology class
    assert len(t["sample"]) == 2
    assert len(t["sample_annotation"]) == 3
    # the tracked instance has both its annotations; the lone object is its own instance
    nbrs = sorted(i["nbr_annotations"] for i in t["instance"])
    assert nbrs == [1, 2]
    # sample next/prev chain
    s0, s1 = t["sample"][0], t["sample"][1]
    assert s0["next"] == s1["token"] and s1["prev"] == s0["token"]
    # 2D box preserved in the non-standard field; 3D is identity placeholder
    ann = t["sample_annotation"][0]
    assert ann["lbx_bbox2d"][:2] == [100.0, 100.0]
    assert ann["size"] == [0.0, 0.0, 0.0]


# --- integration through the export driver -----------------------------------


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")
pytestmark = []


@requires_infra
@pytest.mark.asyncio
async def test_export_openlabel_and_nuscenes_through_driver():
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.export.dataset import SliceSpec, export_dataset, reimport_sanity

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    oid = uuid.uuid4()
    mask_uri = store.put_bytes(
        f"masks/{sid}/{fid}/{oid}.json",
        json.dumps({"encoding": "polygon", "polygons": [[100, 100, 200, 100, 200, 200, 100, 200]],
                    "height": 480, "width": 640}).encode(),
        "application/json",
    )
    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(1),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri=f"s3://x/{fid}.jpg",
                     width=640, height=480, quality=0.9))
        db.add(Object(object_id=oid, frame_id=fid, class_id=6, bbox=[100, 100, 200, 200], conf=0.97,
                      mask_uri=mask_uri, mask_encoding="polygon", attrs={"overload": True},
                      source="auto_accept", state="auto_accept", provenance={"agreement": True}))
        db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=11, bbox=[300, 200, 360, 260], conf=0.96,
                      attrs={}, source="auto_accept", state="auto_accept", provenance={}))
        await db.commit()

    spec = SliceSpec(name="p2-exports", states=["auto_accept"], session_id=str(sid),
                     formats=["openlabel", "nuscenes", "parquet"])
    result = await export_dataset(spec)
    out_dir = Path(result["out_dir"])

    ol = json.loads((out_dir / "openlabel" / "openlabel.json").read_text())["openlabel"]
    assert len(ol["objects"]) == 2
    # masked object has poly2d
    assert any("poly2d" in o["object_data"] for f in ol["frames"].values() for o in f["objects"].values())

    assert (out_dir / "nuscenes" / "sample_annotation.json").exists()
    assert (out_dir / "nuscenes" / "LIMITATIONS.md").exists()

    report = reimport_sanity(out_dir)
    assert report["ok"] is True
    assert report["openlabel_annotations"] == 2
    assert report["nuscenes_annotations"] == 2
