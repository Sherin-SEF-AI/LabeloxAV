"""P1 close-the-loop: regression gate (unit) and the training-set builder (seeded integration)."""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns
from services.training.eval import regression_gate


def test_regression_gate_promotes_on_improvement():
    base = {"map50": 0.40, "per_class": {"autorickshaw": 0.3, "sedan": 0.5}}
    cand = {"map50": 0.46, "per_class": {"autorickshaw": 0.42, "sedan": 0.52}}
    g = regression_gate(cand, base)
    assert g["promote"] is True
    assert g["map50_delta"] == 0.06


def test_regression_gate_rejects_on_map_drop():
    base = {"map50": 0.50, "per_class": {}}
    cand = {"map50": 0.47, "per_class": {}}
    g = regression_gate(cand, base)
    assert g["promote"] is False
    assert any("map50 delta" in r for r in g["reasons"])


def test_regression_gate_ignores_cross_vocabulary_classes():
    # baseline is COCO-named ('bicycle','person'); candidate is ontology-named ('cycle','pedestrian').
    # A class only in the baseline vocabulary is a naming difference, not a regression.
    base = {"map50": 0.05, "per_class": {"bicycle": 0.19, "person": 0.2, "bus": 0.3}}
    cand = {"map50": 0.39, "per_class": {"cycle": 0.25, "pedestrian": 0.31, "bus": 0.55}}
    g = regression_gate(cand, base)
    assert g["promote"] is True
    assert g["regressed_classes"] == []


def test_regression_gate_rejects_on_class_regression():
    base = {"map50": 0.40, "per_class": {"cattle": 0.50, "sedan": 0.40}}
    cand = {"map50": 0.45, "per_class": {"cattle": 0.20, "sedan": 0.42}}  # cattle craters
    g = regression_gate(cand, base, max_class_drop=0.15)
    assert g["promote"] is False
    assert g["regressed_classes"][0]["class"] == "cattle"


# --- builder integration (asyncio_mode=auto runs the async test below) -------


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
async def test_dataset_builder_writes_valid_yolo_dataset():
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.training.dataset_builder import BuildSpec, build_training_dataset

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = now_ns()
    prefix = f"TESTLOOP-{sid.hex[:6]}"

    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(6),
                         city="BLR", route=prefix, sensors={}, ontology_version="labelox-in-0.1.0"))
        await db.flush()
        for i in range(6):
            ts = start + seconds_to_ns(i)
            img = np.random.default_rng(i).integers(20, 230, size=(480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri,
                         width=640, height=480, quality=0.9))
            # two agreement objects per frame (autorickshaw + sedan)
            for cid, box in [(6, [100, 100, 200, 220]), (11, [300, 200, 380, 260])]:
                db.add(Object(frame_id=fid, class_id=cid, bbox=box, conf=0.7, attrs={}, source="fused",
                              state="review", provenance={"agreement": True, "mask_box_disagree": False}))
        await db.commit()

    ds = await build_training_dataset(BuildSpec(name=f"test-{sid.hex[:6]}", route_prefix=prefix,
                                                conf_floor=0.2, max_per_class=100, val_frac=0.34))
    assert ds["classes"] == 2
    assert ds["n_train_images"] >= 1 and ds["n_val_images"] >= 1

    out = Path(ds["dir"])
    assert (out / "data.yaml").exists()
    # a label file parses as YOLO: 5 cols, class idx in range, coords normalized
    label = next((out / "labels/train").glob("*.txt"), None) or next((out / "labels/val").glob("*.txt"))
    for line in label.read_text().splitlines():
        parts = line.split()
        assert len(parts) == 5
        assert 0 <= int(parts[0]) < 2
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
