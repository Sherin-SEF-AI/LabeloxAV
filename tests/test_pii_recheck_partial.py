"""A re-check limited to some targets (plates only, or faces above a stricter score) adds only those, and
never stamps the frame as fully re-checked: the export gate reads that stamp, and a plates-only pass has not
looked for faces."""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


class _Stub:
    available = True

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return list(self.boxes)


class _Anon:
    method_version = "test-anon-v1"

    def __init__(self):
        self.face = _Stub([(100.0, 100.0, 140.0, 150.0, 0.62), (300.0, 100.0, 340.0, 150.0, 0.91)])
        self.plate = _Stub([(500.0, 400.0, 580.0, 430.0, 0.8)])

    def _blur_region(self, img, x1, y1, x2, y2):
        img[int(y1):int(y2), int(x1):int(x2)] = 0


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _frame():
    from core.storage import get_object_store
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
    ok, buf = cv2.imencode(".jpg", np.full((720, 1280, 3), 128, np.uint8))
    uri = store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", buf.tobytes(), "image/jpeg")
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="PII-PART", start_ts_ns=ts, end_ts_ns=ts + 1, city="BLR",
                         sensors={}, ontology_version=get_ontology().version))
        await db.flush()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=1280, height=720,
                     quality=0.9, scene={}))
        await db.commit()
    return fid


async def _check(fid, **kw):
    from core.storage import get_object_store
    from db.models import Frame, PiiAudit
    from db.session import get_sessionmaker
    from services.anonymize.recheck import recheck_frame

    async with get_sessionmaker()() as db:
        frame = await db.get(Frame, fid)
        res = await recheck_frame(db, get_object_store(), _Anon(), frame, **kw)
        await db.commit()
        audit = await db.get(PiiAudit, fid)
        return res, (audit.method_version, [r["type"] for r in audit.regions]) if audit else None


def test_a_plates_only_pass_adds_plates_and_does_not_claim_the_frame_was_fully_checked():
    fid = _run(_frame())
    res, audit = _run(_check(fid, apply=True, types=frozenset({"plate"})))
    assert res["plates_added"] == 1 and res["faces_added"] == 0
    assert audit is not None and audit[1] == ["plate"]
    assert not audit[0].endswith("+roi"), "a plates-only pass stamped the frame as fully re-checked"

    # A stricter faces pass keeps only the confident face, and is still not a full pass.
    res, audit = _run(_check(fid, apply=True, types=frozenset({"face"}), min_scores={"face": 0.75}))
    assert res["faces_added"] == 1 and res["regions_added"][0]["score"] == 0.91
    assert sorted(audit[1]) == ["face", "plate"] and not audit[0].endswith("+roi")

    # A full pass adds what is still missing and stamps it.
    res, audit = _run(_check(fid, apply=True))
    assert res["faces_added"] == 1 and audit[0].endswith("+roi")


def test_a_dry_run_writes_nothing():
    fid = _run(_frame())
    res, audit = _run(_check(fid, apply=False, types=frozenset({"plate"})))
    assert res["status"] == "planned" and res["plates_added"] == 1 and audit is None
