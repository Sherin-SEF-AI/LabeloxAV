"""Two bugs the engine inventory found, held fixed.

1. The keyframe routes (mark a keyframe, re-interpolate around it) imported a function a later commit had
   renamed, so both answered with an ImportError while the editor kept calling them.
2. No flow propagator filtered by camera. In a multi-camera session the next frame by time is usually another
   camera's, so a box was carried from one view into another.
"""

from __future__ import annotations

import asyncio
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db

SIZE, BOX = (360, 480), 60


def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


def _seed(cams: list[str], n: int, keyframes_on: list[int] | None = None) -> dict:
    """A session whose cameras interleave in time; a textured square drifts right on cam_f."""
    from core.storage import get_object_store
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    rng = np.random.default_rng(5)
    texture = rng.integers(0, 255, size=(BOX, BOX), dtype=np.uint8)
    noise = rng.integers(80, 120, size=SIZE, dtype=np.uint8)

    async def _go():
        sid, tid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        cid = onto.by_name("sedan").id
        out: dict = {"session_id": str(sid), "track_id": str(tid), "frames": {c: [] for c in cams}, "objects": []}
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="FLOWCAM", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(n),
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.flush()
            db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts, last_ts_ns=ts + seconds_to_ns(n)))
            await db.flush()
            for i in range(n):
                for k, cam in enumerate(cams):
                    img = noise.copy()
                    if cam == "cam_f":
                        x = 80 + 5 * i
                        img[100:100 + BOX, x:x + BOX] = texture
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                    t = ts + seconds_to_ns(i) // 10 + k * 1000  # interleaved: cam_f, then cam_l, per instant
                    uri = store.put_bytes(f"frames/{sid}/{cam}/{t}.png", buf.tobytes(), "image/png")
                    fid = uuid.uuid4()
                    out["frames"][cam].append(str(fid))
                    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=t, cam_id=cam, img_uri=uri,
                                 width=SIZE[1], height=SIZE[0], quality=0.9, scene={}))
                    await db.flush()
                    if cam == "cam_f" and keyframes_on is not None and i in keyframes_on:
                        oid = uuid.uuid4()
                        out["objects"].append(str(oid))
                        x = 80 + 5 * i
                        db.add(Object(object_id=oid, frame_id=fid, track_id=tid, class_id=cid,
                                      bbox=[float(x), 100.0, float(x + BOX), 160.0], conf=1.0, source="human",
                                      state="accepted", is_keyframe=True, attrs={}, provenance={}, version=1))
            await db.commit()
        return out

    return _run(_go())


async def _frames_of(object_ids: list[str]) -> set[str]:
    from sqlalchemy import select

    from db.models import Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        return {str(f) for f in (await db.execute(select(Object.frame_id).where(
            Object.object_id.in_([uuid.UUID(o) for o in object_ids])))).scalars()}


def test_the_keyframe_routes_work_again():
    from _authutil import auth_headers

    from services.api.main import app

    s = _seed(["cam_f"], 9, keyframes_on=[0, 4, 8])
    with TestClient(app) as c:
        h = auth_headers("reviewer")
        r = c.post(f"/api/objects/{s['objects'][1]}/keyframe", headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["is_keyframe"] is True
        r = c.post(f"/api/objects/{s['objects'][1]}/reinterpolate", headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["track_id"] == s["track_id"] and "created" in r.json()


@pytest.mark.parametrize("which", ["flow", "sam"])
def test_propagation_stays_on_its_own_camera(which):
    from services.intelligence.propagate import propagate_forward
    from services.temporal.sam_propagate import sam_propagate_object

    s = _seed(["cam_f", "cam_l"], 6, keyframes_on=[0])
    src = uuid.UUID(s["objects"][0])
    if which == "flow":
        made = _run(propagate_forward(src, 4))["object_ids"]
    else:
        made = _run(sam_propagate_object(src, 4, "forward", refine=False))["by_direction"]["forward"]
    assert made, "the square moves 5 px a frame on cam_f, so flow should follow it"
    landed = _run(_frames_of(made))
    assert landed <= set(s["frames"]["cam_f"]), "a propagated box landed on another camera's frame"
    assert not landed & set(s["frames"]["cam_l"])
