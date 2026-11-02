"""Propagation measured against accepted tracks: a textured square moves a known distance per frame.

Flow should follow it closely, so the measurement must report a high IoU for the first steps; and a track
whose accepted boxes skip a frame is not a run propagation would see, so it must be left out. SAM refinement
is not exercised here (it needs the model); `refine=False` measures flow alone.
"""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db

SIZE, STEP, BOX = (480, 640), 6, 80


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed(n_frames: int, accepted_on: list[int]) -> str:
    from core.storage import get_object_store
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    rng = np.random.default_rng(11)
    texture = rng.integers(0, 255, size=(BOX, BOX), dtype=np.uint8)
    background = (rng.integers(90, 110, size=SIZE, dtype=np.uint8))

    async def _go():
        sid, ts, tid = uuid.uuid4(), now_ns(), uuid.uuid4()
        cid = onto.by_name("sedan").id
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="PROP-EVAL", start_ts_ns=ts,
                             end_ts_ns=ts + seconds_to_ns(n_frames), city="BLR", sensors={},
                             ontology_version=onto.version))
            await db.flush()
            db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                         last_ts_ns=ts + seconds_to_ns(n_frames)))
            await db.flush()
            for i in range(n_frames):
                x, y = 100 + STEP * i, 150
                img = background.copy()
                img[y:y + BOX, x:x + BOX] = texture
                ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                t = ts + seconds_to_ns(i) // 10
                uri = store.put_bytes(f"frames/{sid}/cam_f/{t}.png", buf.tobytes(), "image/png")
                fid = uuid.uuid4()
                db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t, cam_id="cam_f", img_uri=uri,
                             width=SIZE[1], height=SIZE[0], quality=0.9, scene={}))
                await db.flush()
                if i in accepted_on:
                    db.add(Object(object_id=uuid.uuid4(), frame_id=fid, track_id=tid, class_id=cid,
                                  bbox=[float(x), float(y), float(x + BOX), float(y + BOX)], conf=1.0,
                                  source="human", state="accepted", attrs={}, provenance={}, version=1))
            await db.commit()
        return str(tid)

    return _run(_go())


def test_flow_follows_a_moving_object_on_an_accepted_track():
    from db.session import get_sessionmaker
    from services.temporal.propagation_eval import evaluate_propagation

    tid = _seed(5, accepted_on=[0, 1, 2, 3, 4])

    async def _go():
        async with get_sessionmaker()() as db:
            return await evaluate_propagation(db, steps=3, refine=False, track_ids=[uuid.UUID(tid)])

    out = _run(_go())
    assert out["tracks"] == 1 and "flow_sam" not in out["methods"]
    steps = out["methods"]["flow"]
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert all(s["lost"] == 0 for s in steps)
    # A rigid textured square moving 6 px a frame: flow should land almost exactly on it.
    assert steps[0]["mean_iou"] >= 0.85 and steps[0]["keep"] == 1.0, steps


def test_a_track_with_a_gap_is_not_a_run():
    from db.session import get_sessionmaker
    from services.temporal.propagation_eval import accepted_runs

    # Accepted on frames 0, 2, 3, 4: no three steps in a row from any start.
    tid = _seed(5, accepted_on=[0, 2, 3, 4])

    async def _go():
        async with get_sessionmaker()() as db:
            return await accepted_runs(db, steps=3, n_tracks=10, track_ids=[uuid.UUID(tid)])

    assert _run(_go()) == []
