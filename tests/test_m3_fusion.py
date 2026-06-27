"""M3 tests: fusion matching/voting/geometry, calibration, gate states (pure unit, no GPU), the
provenance walk (DB), and the full autolabel pipeline (GPU)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.schemas import GateState
from services.autolabel.fusion import FusionEngine
from services.autolabel.gate import gate_object, needs_vlm
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.base import RawDetection


def _mask(h, w, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _engine() -> FusionEngine:
    return FusionEngine(get_settings(), get_ontology())


def test_overlapping_detections_fuse_with_agreement():
    eng = _engine()
    fid = uuid.uuid4()
    a = RawDetection("path_a_yolo26", (100, 100, 200, 200), 0.97, "yolo11l.pt", "sedan", 11)
    b = RawDetection(
        "path_b_sam3", (104, 104, 204, 204), 0.95, "world+sam", "sedan", 11, mask=_mask(300, 300, 104, 104, 204, 204)
    )
    fused = eng.fuse_frame(fid, [a], [b])
    assert len(fused) == 1
    fo = fused[0]
    assert fo.obj.class_name == "sedan"
    assert fo.obj.provenance.agreement is True
    assert fo.mask is not None  # SAM mask preferred for geometry
    assert fo.obj.conf >= 0.95
    state = gate_object(fo.obj, get_ontology(), get_settings().gate)
    assert state == GateState.auto_accept  # high conf, agreement, not rare


def test_non_overlapping_detections_are_singletons_not_autoaccept():
    eng = _engine()
    fid = uuid.uuid4()
    a = RawDetection("path_a_yolo26", (10, 10, 50, 50), 0.99, "yolo11l.pt", "sedan", 11)
    b = RawDetection("path_b_sam3", (500, 500, 560, 560), 0.99, "world+sam", "truck", 19, mask=_mask(700, 700, 500, 500, 560, 560))
    fused = eng.fuse_frame(fid, [a], [b])
    assert len(fused) == 2
    for fo in fused:
        # single-path: no consensus, never auto-accept
        assert fo.obj.provenance.agreement is False
        assert gate_object(fo.obj, get_ontology(), get_settings().gate) != GateState.auto_accept


def test_class_disagreement_penalizes_confidence():
    eng = _engine()
    fid = uuid.uuid4()
    a = RawDetection("path_a_yolo26", (100, 100, 200, 200), 0.95, "yolo11l.pt", "sedan", 11)
    b = RawDetection("path_b_sam3", (102, 102, 202, 202), 0.95, "world+sam", "truck", 19, mask=_mask(300, 300, 102, 102, 202, 202))
    fused = eng.fuse_frame(fid, [a], [b])
    assert len(fused) == 1
    fo = fused[0]
    assert fo.obj.provenance.agreement is False
    assert any(p.verdict == "overruled" for p in fo.obj.provenance.proposals)


def test_rare_class_forces_review_even_at_high_confidence():
    eng = _engine()
    fid = uuid.uuid4()
    # autorickshaw is India-specific (rare): agreement + high conf must still route to review.
    a = RawDetection("path_a_yolo26", (100, 100, 200, 200), 0.98, "yolo11l.pt", "autorickshaw", 6)
    b = RawDetection("path_b_sam3", (101, 101, 201, 201), 0.98, "world+sam", "autorickshaw", 6, mask=_mask(300, 300, 101, 101, 201, 201))
    fo = eng.fuse_frame(fid, [a], [b])[0]
    assert gate_object(fo.obj, get_ontology(), get_settings().gate) == GateState.review
    assert needs_vlm(fo.obj, get_ontology(), get_settings().gate) is True


def test_low_confidence_routes_to_annotate():
    eng = _engine()
    fid = uuid.uuid4()
    a = RawDetection("path_a_yolo26", (100, 100, 130, 130), 0.30, "yolo11l.pt", "object_fallback", 45)
    fo = eng.fuse_frame(fid, [a], [])[0]
    assert gate_object(fo.obj, get_ontology(), get_settings().gate) == GateState.annotate


# --- DB: provenance walk -----------------------------------------------------
# (asyncio_mode=auto runs the async tests below; no module-level mark needed since this file
# mixes sync unit tests with async DB/GPU tests.)


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
async def test_provenance_walk_returns_complete_chain():
    from core.provenance import walk_provenance
    from core.timebase import now_ns
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    sid = uuid.uuid4()
    fid = uuid.uuid4()
    oid = uuid.uuid4()
    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-07", start_ts_ns=now_ns(), end_ts_ns=now_ns(),
                         city="BLR", sensors={"cam_f": {"serial": "s", "calibration_hash": "h"}},
                         raw_uri="s3://x/raw.mp4", ontology_version="labelox-in-0.1.0"))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=now_ns(), cam_id="cam_f",
                     img_uri="s3://x/f.jpg", width=640, height=480, quality=0.8))
        db.add(Object(object_id=oid, frame_id=fid, class_id=6, bbox=[1, 2, 3, 4], conf=0.7,
                      source="fused", state="review",
                      provenance={"proposals": [{"path": "path_a_yolo26", "model_version": "yolo11l.pt",
                                                 "verdict": "agree"}], "agreement": False}))
        await db.commit()

        chain = await walk_provenance(db, oid)
        assert chain["frame"]["frame_id"] == str(fid)
        assert chain["session"]["raw_uri"] == "s3://x/raw.mp4"
        assert chain["session"]["sensors"]["cam_f"]["calibration_hash"] == "h"
        assert "yolo11l.pt" in chain["model_versions"]


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not (_cuda_ready() and _infra_up()), reason="needs GPU + infra")


@requires_gpu
async def test_autolabel_pipeline_writes_gated_objects():
    import cv2

    from core.storage import get_object_store
    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.runner import autolabel_session
    from sqlalchemy import func, select

    store = get_object_store()
    store.ensure_bucket()
    maker = get_sessionmaker()
    sid = uuid.uuid4()
    start = now_ns()
    rng = np.random.default_rng(5)
    async with maker() as db:
        db.add(DbSession(session_id=sid, vehicle_id="TIGOR-T", start_ts_ns=start, end_ts_ns=start + seconds_to_ns(2),
                         city="BLR", sensors={}, ontology_version="labelox-in-0.1.0"))
        await db.flush()
        for i in range(2):
            ts = start + seconds_to_ns(i)
            img = rng.integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
            db.add(Frame(session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=640, height=480, quality=0.8))
        await db.commit()

    summary = await autolabel_session(sid, limit=2)
    assert summary["frames"] == 2
    assert summary["peak_vram_mb"] <= summary["vram_ceiling_mb"]

    maker2 = get_sessionmaker()
    async with maker2() as db:
        n = (await db.execute(select(func.count()).select_from(Object).join(Frame).where(Frame.session_id == sid))).scalar_one()
        assert n == summary["objects"]
