"""Event cameras (#10): readers on fixtures written in the published layouts, accumulation, and frames to label.

No event-camera recording exists on this machine. The fixtures are an .npy as Prophesee-style tools save it
(structured x, y, t, p) and as a plain (N, 4) array, and an MCAP with `dvs_msgs/EventArray` messages in ROS 2 CDR
and in ROS 1, so the CDR alignment of the Event struct (13 bytes, then a 16-byte stride) is exercised.
"""

from __future__ import annotations

import asyncio
import io
import uuid

import numpy as np
import pytest

from services.sensors.events import (
    EventFormatError,
    accumulate,
    count_image,
    encode_event_array_cdr,
    encode_event_array_ros1,
    parse_event_array_cdr,
    parse_event_array_ros1,
    read_npy,
    time_surface,
)


def _events(n=2000, w=64, h=48, t0_us=1_000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(t0_us, t0_us + 100_000, n))
    return rng.integers(0, w, n), rng.integers(0, h, n), t, rng.integers(0, 2, n)


def test_structured_npy_reads_with_unit_and_polarity_conventions():
    x, y, t, p = _events()
    arr = np.zeros(len(x), dtype=[("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    arr["x"], arr["y"], arr["t"], arr["p"] = x, y, t, np.where(p > 0, 1, -1)   # -1/1 polarity
    buf = io.BytesIO()
    np.save(buf, arr)
    ev = read_npy(buf.getvalue(), width=64, height=48)
    assert ev.n == len(x) and (ev.width, ev.height) == (64, 48)
    assert ev.t[0] == int(t[0]) * 1000 and np.all(np.diff(ev.t) >= 0)       # microseconds -> ns, sorted
    assert set(np.unique(ev.p)) <= {0, 1} and int(ev.p.sum()) == int((p > 0).sum())


def test_plain_npy_with_a_column_order_and_bad_inputs():
    x, y, t, p = _events(100)
    arr = np.stack([t, x, y, p], axis=1).astype(np.float64)
    buf = io.BytesIO()
    np.save(buf, arr)
    ev = read_npy(buf.getvalue(), columns="t,x,y,p", time_unit="us")
    assert ev.width == int(x.max()) + 1 and ev.height == int(y.max()) + 1     # inferred when not given
    with pytest.raises(EventFormatError, match="outside"):
        read_npy(buf.getvalue(), columns="t,x,y,p", width=4, height=4)
    with pytest.raises(EventFormatError, match="columns"):
        read_npy(buf.getvalue(), columns="t,x,x,p")


def test_event_array_cdr_and_ros1_round_trip():
    x, y, t, p = _events(37)
    t_ns = t * 1000 + 1_700_000_000_000_000_000
    msg = encode_event_array_cdr(x, y, t_ns, p, width=64, height=48, stamp_ns=int(t_ns[0]), frame_id="cam0")
    w, h, rec = parse_event_array_cdr(msg)
    assert (w, h) == (64, 48)
    assert rec["x"].tolist() == x.tolist() and rec["y"].tolist() == y.tolist()
    assert (rec["sec"] * 1_000_000_000 + rec["nsec"]).tolist() == t_ns.tolist()
    assert rec["p"].tolist() == p.tolist()
    msg1 = encode_event_array_ros1(x, y, t_ns, p, width=64, height=48, stamp_ns=int(t_ns[0]))
    w1, h1, rec1 = parse_event_array_ros1(msg1)
    assert (w1, h1) == (64, 48) and rec1["x"].tolist() == x.tolist() and rec1["p"].tolist() == p.tolist()
    # one event and none at all are edge cases of the stride
    assert parse_event_array_cdr(encode_event_array_cdr(x[:1], y[:1], t_ns[:1], p[:1], width=64, height=48,
                                                        stamp_ns=0))[2]["x"].tolist() == [int(x[0])]
    assert len(parse_event_array_cdr(encode_event_array_cdr([], [], [], [], width=8, height=8, stamp_ns=0))[2]) == 0


def _mcap(path, x, y, t_ns, p, encoding="cdr", chunks=3):
    from mcap.writer import Writer

    name = "dvs_msgs/msg/EventArray" if encoding == "cdr" else "dvs_msgs/EventArray"
    enc = encode_event_array_cdr if encoding == "cdr" else encode_event_array_ros1
    with open(path, "wb") as f:
        w = Writer(f)
        w.start()
        sc = w.register_schema(name=name, encoding="ros2msg" if encoding == "cdr" else "ros1msg", data=b"")
        ch = w.register_channel(topic="/dvs/events", message_encoding=encoding, schema_id=sc)
        for part in np.array_split(np.arange(len(x)), chunks):
            ts = int(t_ns[part[0]])
            w.add_message(ch, log_time=ts, publish_time=ts,
                          data=enc(x[part], y[part], t_ns[part], p[part], width=64, height=48, stamp_ns=ts))
        w.finish()


@pytest.mark.parametrize("encoding", ["cdr", "ros1"])
def test_mcap_event_arrays_concatenate_in_time_order(tmp_path, encoding):
    from services.sensors.events import read_mcap

    x, y, t, p = _events(500)
    t_ns = t * 1000
    path = tmp_path / f"ev_{encoding}.mcap"
    _mcap(path, x, y, t_ns, p, encoding)
    ev = read_mcap(path)
    assert ev.n == 500 and (ev.width, ev.height) == (64, 48)
    assert ev.t.tolist() == sorted(t_ns.tolist()) and int(ev.p.sum()) == int(p.sum())


def test_accumulation_counts_and_time_surface():
    from services.sensors.events import Events

    # 3 positive events at (1, 1) and 1 negative at (2, 2) in the first 10 ms; one at (3, 3) at 25 ms.
    ev = Events(x=np.array([1, 1, 1, 2, 3], np.uint16), y=np.array([1, 1, 1, 2, 3], np.uint16),
                t=np.array([0, 1, 2, 5, 25], np.int64) * 1_000_000, p=np.array([1, 1, 1, 0, 1], np.uint8),
                width=8, height=6)
    img = count_image(ev, 0, 4)
    assert img.shape == (6, 8, 3)
    assert img[1, 1, 0] > img[1, 1, 2] and img[2, 2, 2] > img[2, 2, 0] and img[0, 0].tolist() == [24, 24, 24]
    ts = time_surface(ev, 0, 4, end := 10_000_000, tau_ns=5e6)
    # the (2, 2) event is 5 ms old, the (1, 1) ones 8 ms: the newer is brighter in its own channel
    assert ts[2, 2, 2] > ts[1, 1, 0] > 24
    frames = accumulate(ev, mode="count", window_ms=10)
    assert [f["n_events"] for f in frames] == [4, 1] and frames[0]["ts_ns"] == end
    assert len(accumulate(ev, mode="count", window_ms=10, max_frames=1)) == 1
    with pytest.raises(ValueError):
        accumulate(ev, mode="voxel")


# ---- database ----------------------------------------------------------------------------------------------

def _run(coro):
    from _authutil import _clear_db_cache

    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


@pytest.mark.db
def test_import_accumulate_label_and_replace(tmp_path):
    """An MCAP becomes a stream on a new session; accumulation writes real, tagged frames; a labelled frame
    survives a re-accumulation and an unlabelled one is replaced."""
    from sqlalchemy import select

    from core.origin import REAL
    from core.storage import get_object_store
    from db.models import EventStream, Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.sensors.events import accumulate_stream, import_events

    x, y, t, p = _events(3000)
    t_ns = t * 1000 + 1_700_000_000_000_000_000
    path = tmp_path / "ev.mcap"
    _mcap(path, x, y, t_ns, p)
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.mcap", path.read_bytes())
    res = _run(import_events({"source_uri": uri, "sensor": "event_front",
                              "new_session": {"vehicle_id": "EVT-T", "city": "BLR"}}))
    assert res["events"] == 3000 and (res["width"], res["height"]) == (64, 48)
    stream_id = uuid.UUID(res["stream_id"])
    acc = _run(accumulate_stream(stream_id, mode="count", window_ms=20))
    assert acc["frames"] == 5 and acc["removed"] == 0

    async def label_one():
        async with get_sessionmaker()() as db:
            frames = (await db.execute(select(Frame).where(Frame.session_id == uuid.UUID(res["session_id"]))
                                       .order_by(Frame.ts_ns))).scalars().all()
            assert all(f.origin == REAL and f.cam_id == "event_front" and "sensor:event" in f.tags for f in frames)
            assert all((f.width, f.height) == (64, 48) for f in frames)
            sess = await db.get(DbSession, uuid.UUID(res["session_id"]))
            assert sess.origin == REAL and sess.vehicle_id == "EVT-T"
            db.add(Object(frame_id=frames[0].frame_id, class_id=get_ontology().by_name("pedestrian").id,
                          bbox=[1, 1, 10, 10], conf=1.0, source="human", state="accepted"))
            await db.commit()
            return frames[0].frame_id

    labelled = _run(label_one())
    acc2 = _run(accumulate_stream(stream_id, mode="time_surface", window_ms=50, tau_ms=10))
    assert acc2["kept_labelled"] == 1 and acc2["removed"] == 4 and acc2["frames"] == 2

    async def after():
        async with get_sessionmaker()() as db:
            frames = (await db.execute(select(Frame).where(Frame.session_id == uuid.UUID(res["session_id"]))
                                       )).scalars().all()
            st = await db.get(EventStream, stream_id)
            return frames, st

    frames, st = _run(after())
    assert labelled in {f.frame_id for f in frames} and len(frames) == 3
    assert st.accumulation["mode"] == "time_surface" and st.stats["positive"] + st.stats["negative"] == 3000


@pytest.mark.db
def test_event_routes(tmp_path):
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from core.storage import get_object_store
    from services.api.main import app
    from services.tools.registry import run_job

    x, y, t, p = _events(800)
    arr = np.zeros(len(x), dtype=[("x", "<u2"), ("y", "<u2"), ("t", "<i8"), ("p", "u1")])
    arr["x"], arr["y"], arr["t"], arr["p"] = x, y, t + 1_700_000_000_000_000, p
    buf = io.BytesIO()
    np.save(buf, arr)
    uri = get_object_store().put_bytes(f"uploads/test/{uuid.uuid4()}.npy", buf.getvalue())
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        body = {"source_uri": uri, "sensor": "dvs", "width": 64, "height": 48}
        assert c.post("/api/tools/sensors/events/import", headers=ann, json=body).status_code == 403
        assert c.post("/api/tools/sensors/events/import", headers=rev,
                      json={**body, "sensor": "../x"}).status_code == 422
        r = c.post("/api/tools/sensors/events/import", headers=rev, json=body)
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
    _run(run_job(uuid.UUID(jid)))
    with TestClient(app) as c:
        ann, rev = auth_headers("annotator"), auth_headers("reviewer")
        j = c.get(f"/api/tools/jobs/{jid}", headers=ann).json()
        assert j["status"] == "done", j
        sid = j["result"]["stream_id"]
        pv = c.get(f"/api/tools/sensors/events/streams/{sid}/preview.jpg?window_ms=40", headers=ann)
        assert pv.status_code == 200 and pv.headers["content-type"] == "image/jpeg" and int(pv.headers["X-Events"]) > 0
        r = c.post(f"/api/tools/sensors/events/streams/{sid}/accumulate", headers=rev,
                   json={"mode": "count", "window_ms": 25})
        assert r.status_code == 200
        jid2 = r.json()["job_id"]
    _run(run_job(uuid.UUID(jid2)))
    with TestClient(app) as c:
        ann = auth_headers("annotator")
        fr = c.get(f"/api/tools/sensors/events/streams/{sid}/frames", headers=ann).json()
        assert len(fr["frames"]) == 4 and fr["accumulation"]["frames"] == 4
        img = c.get(f"/api/frames/{fr['frames'][0]['frame_id']}/image", headers=ann)
        assert img.status_code == 200 and len(img.content) > 100
        ov = c.get("/api/tools/sensors/overview", headers=ann).json()
        assert any(e["stream_id"] == sid for e in ov["events"])
