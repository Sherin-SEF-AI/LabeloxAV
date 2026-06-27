"""M5: seal a dataset commit and export COCO + YOLO + Parquet sidecar, then reimport-sanity check.
Requires infra (DB + MinIO). No GPU."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.asyncio


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


async def _seed_objects() -> uuid.UUID:
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()

    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f",
                     img_uri=f"s3://x/frames/{sid}/cam_f/{start}.jpg", width=640, height=480, quality=0.9))
        # one masked autorickshaw, one box-only sedan, both auto_accept
        oid = uuid.uuid4()
        mask_uri = store.put_bytes(
            f"masks/{sid}/{fid}/{oid}.json",
            json.dumps({"encoding": "polygon", "polygons": [[100, 100, 200, 100, 200, 200, 100, 200]],
                        "height": 480, "width": 640}).encode(),
            "application/json",
        )
        db.add(Object(object_id=oid, frame_id=fid, class_id=6, bbox=[100, 100, 200, 200], conf=0.97,
                      mask_uri=mask_uri, mask_encoding="polygon", attrs={"overload": True},
                      source="auto_accept", state="auto_accept",
                      provenance={"proposals": [{"path": "path_a_yolo26", "model_version": "yolo11l.pt", "verdict": "agree"}],
                                  "agreement": True}))
        db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=11, bbox=[300, 200, 360, 260], conf=0.96,
                      attrs={}, source="auto_accept", state="auto_accept",
                      provenance={"proposals": [], "agreement": True}))
        await db.commit()
    return sid


@requires_infra
async def test_export_seals_commit_and_passes_reimport():
    from services.export.dataset import SliceSpec, export_dataset, reimport_sanity
    from db.models import DatasetCommit
    from db.session import get_sessionmaker

    sid = await _seed_objects()
    spec = SliceSpec(
        name="m5-demo", states=["auto_accept"], session_id=str(sid),
        formats=["coco", "yolo", "parquet"],
    )
    result = await export_dataset(spec)

    assert result["object_count"] == 2
    assert result["commit_id"].startswith("lbx-")

    out_dir = Path(result["out_dir"])
    # COCO has polygon segmentation on the masked object
    coco = json.loads((out_dir / "coco" / "annotations.json").read_text())
    assert len(coco["annotations"]) == 2
    assert any(len(a["segmentation"]) > 0 for a in coco["annotations"])
    # extension block preserves attributes COCO cannot natively carry
    assert any(a["labelox"]["attributes"].get("overload") is True for a in coco["annotations"])
    # YOLO labels written
    assert (out_dir / "data.yaml").exists()
    assert list((out_dir / "labels").glob("*.txt"))

    report = reimport_sanity(out_dir)
    assert report["ok"] is True
    assert report["parquet_rows"] == 2
    assert report["coco_annotations"] == 2

    # dataset_commit row sealed and immutable id recorded
    maker = get_sessionmaker()
    async with maker() as db:
        commit = await db.get(DatasetCommit, result["commit_id"])
        assert commit is not None
        assert commit.object_count == 2
        assert commit.ontology_version == "labelox-in-0.1.0"


@requires_infra
async def test_commit_id_is_deterministic_for_same_slice():
    from services.export.dataset import SliceSpec, fetch_records, seal_commit_id

    sid = await _seed_objects()
    spec = SliceSpec(name="determinism", states=["auto_accept"], session_id=str(sid))
    recs = await fetch_records(spec)
    a = seal_commit_id(spec, recs, "labelox-in-0.1.0")
    b = seal_commit_id(spec, recs, "labelox-in-0.1.0")
    assert a == b
