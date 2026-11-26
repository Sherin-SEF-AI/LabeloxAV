"""Event cameras: read (x, y, t, polarity) streams, keep them, and accumulate them into frames people label.

An event camera reports a pixel's brightness change when it happens, not frames at a rate. A person cannot
draw a box on a stream of events, so the stream is accumulated over a window into an image, and that image is
an ordinary `frame` (origin real: the pixels are a measurement, not a composite) with the sensor as its cam_id
and a `sensor:event` tag. Everything the editor, review and export do with frames then works unchanged.

Formats, each by its published layout:

  .npy    a structured array with x, y, t (or ts / timestamp), p (or polarity) fields, as Prophesee's
          tooling and most research code save it; or a plain (N, 4) array whose column order is given
          (default x, y, t, p). Time unit given (default microseconds).
  .h5     DSEC: /events/{x, y, t, p} with t in microseconds after /t_offset; also a single (N, 4) `events`
          dataset. Needs h5py, which is not installed on this machine: the reader says so.
  MCAP    dvs_msgs/EventArray and prophesee_event_msgs/EventArray, ROS 1 or ROS 2 (CDR): Header, uint32
          height, uint32 width, Event[] with uint16 x, uint16 y, time ts, bool polarity.

Accumulation modes:
  count          per pixel, positive and negative event counts in the window (red / blue), normalised by
                 the 99th percentile so one hot pixel does not black out the frame.
  time_surface   per pixel and polarity, exp(-(t_end - t_last) / tau): what moved most recently is bright.

There is no event-camera data on this machine; this is tested on fixtures written in these formats.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import struct
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.logging import get_logger
from core.origin import REAL

log = get_logger("sensors.events")

ProgressFn = Callable[[float, str], Awaitable[None]]
MAX_EVENTS = 60_000_000
EVENT_ARRAY_SCHEMAS = ("dvs_msgs/msg/EventArray", "dvs_msgs/EventArray", "prophesee_event_msgs/msg/EventArray",
                       "prophesee_event_msgs/EventArray")
TIME_UNITS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
EVENT_TAG = "sensor:event"


class EventFormatError(ValueError):
    """The bytes are not an event stream this reader can interpret."""


@dataclass
class Events:
    x: np.ndarray        # uint16
    y: np.ndarray        # uint16
    t: np.ndarray        # int64 ns, sorted
    p: np.ndarray        # uint8, 1 = brightness up, 0 = down
    width: int
    height: int

    @property
    def n(self) -> int:
        return int(len(self.t))

    def sorted(self) -> Events:
        if self.n and np.any(np.diff(self.t) < 0):
            o = np.argsort(self.t, kind="stable")
            return Events(self.x[o], self.y[o], self.t[o], self.p[o], self.width, self.height)
        return self

    def to_npz_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, x=self.x.astype(np.uint16), y=self.y.astype(np.uint16), t=self.t.astype(np.int64),
                            p=self.p.astype(np.uint8), size=np.array([self.width, self.height], np.int32))
        return buf.getvalue()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> Events:
        z = np.load(io.BytesIO(data))
        w, h = (int(v) for v in z["size"])
        return cls(z["x"], z["y"], z["t"], z["p"], w, h)


def _make(x, y, t, p, *, width: int | None, height: int | None, unit_ns: int, offset_ns: int = 0) -> Events:
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    if len(x) == 0:
        raise EventFormatError("the stream has no events")
    if x.min() < 0 or y.min() < 0:
        raise EventFormatError("negative pixel coordinates")
    w = int(width or (int(x.max()) + 1))
    h = int(height or (int(y.max()) + 1))
    if x.max() >= w or y.max() >= h:
        raise EventFormatError(f"events reach ({int(x.max())}, {int(y.max())}) outside the {w}x{h} sensor")
    tt = np.asarray(t)
    t_ns = (tt.astype(np.float64) * unit_ns).round().astype(np.int64) if tt.dtype.kind == "f" \
        else tt.astype(np.int64) * int(unit_ns)
    pol = np.asarray(p)
    pol = (pol > 0).astype(np.uint8)                      # 1/0, 1/-1 and True/False all mean up/down
    return Events(x.astype(np.uint16), y.astype(np.uint16), t_ns + int(offset_ns), pol, w, h).sorted()


def read_npy(data: bytes, *, columns: str = "x,y,t,p", time_unit: str = "us", width: int | None = None,
             height: int | None = None) -> Events:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    unit = TIME_UNITS.get(time_unit)
    if unit is None:
        raise EventFormatError(f"time_unit must be one of {sorted(TIME_UNITS)}")
    if arr.dtype.names:
        names = {n.lower(): n for n in arr.dtype.names}

        def pick(*opts):
            for o in opts:
                if o in names:
                    return arr[names[o]]
            raise EventFormatError(f"structured events need {opts[0]}; fields are {list(arr.dtype.names)}")
        return _make(pick("x"), pick("y"), pick("t", "ts", "timestamp", "time"), pick("p", "polarity", "pol"),
                     width=width, height=height, unit_ns=unit)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise EventFormatError(f"a plain event array must be (N, 4); this one is {arr.shape}")
    cols = [c.strip().lower() for c in columns.split(",")]
    if sorted(cols) != ["p", "t", "x", "y"]:
        raise EventFormatError("columns must name x, y, t and p once each")
    c = {name: arr[:, i] for i, name in enumerate(cols)}
    return _make(c["x"], c["y"], c["t"], c["p"], width=width, height=height, unit_ns=unit)


def read_h5(path: Path, *, width: int | None = None, height: int | None = None) -> Events:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise EventFormatError("reading .h5 event files needs the h5py package, which is not installed on this "
                               "server; convert to .npy or ask an operator to install h5py") from exc
    with h5py.File(path, "r") as f:
        if "events" in f and isinstance(f["events"], h5py.Group):
            g = f["events"]
            off = int(np.asarray(f["t_offset"])) if "t_offset" in f else 0
            return _make(g["x"][:], g["y"][:], g["t"][:], g["p"][:], width=width, height=height, unit_ns=1_000,
                         offset_ns=off * 1_000)
        if "events" in f:
            a = f["events"][:]
            return _make(a[:, 0], a[:, 1], a[:, 2], a[:, 3], width=width, height=height, unit_ns=1_000)
    raise EventFormatError("no /events group or dataset in the .h5 (DSEC layout expected)")


# --- MCAP EventArray --------------------------------------------------------------------------------------

_E0 = np.dtype({"names": ["x", "y", "sec", "nsec", "p"], "formats": ["u2", "u2", "i4", "u4", "u1"],
                "offsets": [0, 2, 4, 8, 12], "itemsize": 13})
_EN = np.dtype({"names": ["x", "y", "sec", "nsec", "p"], "formats": ["u2", "u2", "i4", "u4", "u1"],
                "offsets": [0, 2, 6, 10, 14], "itemsize": 16})
_ROS1 = np.dtype({"names": ["x", "y", "sec", "nsec", "p"], "formats": ["<u2", "<u2", "<u4", "<u4", "u1"],
                  "offsets": [0, 2, 4, 8, 12], "itemsize": 13})


def _cdr_header(buf: bytes) -> tuple[str, int]:
    if len(buf) < 4 or buf[:2] not in (b"\x00\x00", b"\x00\x01"):
        raise EventFormatError("not a CDR message")
    return ("<" if buf[:2] == b"\x00\x01" else ">"), 4


def parse_event_array_cdr(buf: bytes) -> tuple[int, int, np.ndarray]:
    """A ROS 2 EventArray: (width, height, records with x, y, sec, nsec, p).

    CDR aligns every primitive to its own size from the start of the body. The sequence starts 4-aligned, so
    the first Event's int32 needs no padding (13 bytes to its polarity), and every later one starts 2 bytes past
    a 4-boundary, which puts 2 bytes of padding before its seconds: a fixed 16-byte stride from the second on.
    """
    order, base = _cdr_header(buf)
    pos = base

    def u32(p):
        p += (-(p - base)) % 4
        return struct.unpack_from(order + "I", buf, p)[0], p + 4
    pos += 8                                              # stamp: int32 sec, uint32 nanosec
    n_name, pos = u32(pos)
    pos += n_name                                         # frame_id with its NUL
    height, pos = u32(pos)
    width, pos = u32(pos)
    n, pos = u32(pos)
    if n == 0:
        return width, height, np.zeros(0, dtype=_E0)
    if len(buf) < pos + 13 + 16 * (n - 1):
        raise EventFormatError(f"EventArray says {n} events but the message ends at {len(buf)} bytes")
    # The last Event has no trailing padding, so its 16-byte record would read one byte past the end.
    padded = bytes(buf) + b"\x00\x00"
    e0 = np.frombuffer(padded, dtype=_E0.newbyteorder(order), count=1, offset=pos)
    rest = np.frombuffer(padded, dtype=_EN.newbyteorder(order), count=n - 1, offset=pos + 14) if n > 1 else None
    out = np.empty(n, dtype=[("x", "u2"), ("y", "u2"), ("sec", "i8"), ("nsec", "i8"), ("p", "u1")])
    for k in ("x", "y", "sec", "nsec", "p"):
        out[k][:1] = e0[k]
        if rest is not None:
            out[k][1:] = rest[k]
    return width, height, out


def parse_event_array_ros1(buf: bytes) -> tuple[int, int, np.ndarray]:
    """A ROS 1 EventArray: little-endian, packed, Header with seq, 13 bytes per Event."""
    pos = 4 + 8                                           # seq, stamp
    (n_name,) = struct.unpack_from("<I", buf, pos)
    pos += 4 + n_name
    height, width, n = struct.unpack_from("<III", buf, pos)
    pos += 12
    if len(buf) < pos + 13 * n:
        raise EventFormatError(f"EventArray says {n} events but the message ends at {len(buf)} bytes")
    return width, height, np.frombuffer(buf, dtype=_ROS1, count=n, offset=pos)


def _cdr_writer(order: str = "<"):
    body = bytearray()

    def put(fmt: str, *vals):
        size = struct.calcsize(order + fmt[0])
        body.extend(b"\x00" * ((-len(body)) % size))
        body.extend(struct.pack(order + fmt, *vals))
    return body, put


def encode_event_array_cdr(x, y, t_ns, p, *, width: int, height: int, stamp_ns: int, frame_id: str = "event") -> bytes:
    """Serialize events as a ROS 2 EventArray (CDR, little-endian), primitive by primitive under CDR's
    alignment rule. The writer side of `parse_event_array_cdr`, for fixtures and round-trip tests."""
    body, put = _cdr_writer("<")
    put("i", int(stamp_ns // 1_000_000_000))
    put("I", int(stamp_ns % 1_000_000_000))
    name = frame_id.encode() + b"\x00"
    put("I", len(name))
    body.extend(name)
    put("I", int(height))
    put("I", int(width))
    put("I", len(x))
    for xi, yi, ti, pi in zip(x, y, t_ns, p, strict=True):
        put("H", int(xi))
        put("H", int(yi))
        put("i", int(ti) // 1_000_000_000)
        put("I", int(ti) % 1_000_000_000)
        put("B", 1 if pi else 0)
    return b"\x00\x01\x00\x00" + bytes(body)


def encode_event_array_ros1(x, y, t_ns, p, *, width: int, height: int, stamp_ns: int, frame_id: str = "event") -> bytes:
    name = frame_id.encode()
    out = bytearray(struct.pack("<III", 0, stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000))
    out += struct.pack("<I", len(name)) + name + struct.pack("<III", height, width, len(x))
    for xi, yi, ti, pi in zip(x, y, t_ns, p, strict=True):
        out += struct.pack("<HHIIB", int(xi), int(yi), int(ti) // 1_000_000_000, int(ti) % 1_000_000_000,
                           1 if pi else 0)
    return bytes(out)


def read_mcap(path: Path, *, topics: list[str] | None = None) -> Events:
    """Every EventArray message of the file (or of `topics`), concatenated in time order."""
    from mcap.reader import make_reader

    xs, ys, ts, ps = [], [], [], []
    size = None
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(topics=topics):
            if schema is None or schema.name not in EVENT_ARRAY_SCHEMAS:
                continue
            enc = (channel.message_encoding or "").lower()
            if enc == "cdr":
                w, h, rec = parse_event_array_cdr(message.data)
            elif enc == "ros1":
                w, h, rec = parse_event_array_ros1(message.data)
            else:
                continue
            size = size or (int(w), int(h))
            xs.append(rec["x"])
            ys.append(rec["y"])
            ts.append(rec["sec"].astype(np.int64) * 1_000_000_000 + rec["nsec"].astype(np.int64))
            ps.append(rec["p"])
    if not xs:
        raise EventFormatError("the MCAP has no dvs_msgs or prophesee_event_msgs EventArray channel")
    return _make(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(ps),
                 width=size[0] or None, height=size[1] or None, unit_ns=1)


def read_events(path: Path, fmt: str | None = None, **kw) -> Events:
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "npy":
        return read_npy(path.read_bytes(), columns=kw.get("columns") or "x,y,t,p",
                        time_unit=kw.get("time_unit") or "us", width=kw.get("width"), height=kw.get("height"))
    if fmt in ("h5", "hdf5"):
        return read_h5(path, width=kw.get("width"), height=kw.get("height"))
    if fmt == "mcap":
        return read_mcap(path, topics=kw.get("topics"))
    raise EventFormatError(f"unknown event format {fmt!r} (npy, h5, mcap)")


# --- accumulation -----------------------------------------------------------------------------------------

def window_bounds(t: np.ndarray, t0: int, t1: int, window_ns: int) -> list[tuple[int, int, int, int]]:
    """(start_ns, end_ns, first index, last index) of each window over [t0, t1]."""
    starts = np.arange(int(t0), int(t1) + 1, int(window_ns), dtype=np.int64)
    ends = starts + int(window_ns)
    a = np.searchsorted(t, starts, side="left")
    b = np.searchsorted(t, ends, side="left")
    return [(int(s), int(e), int(i), int(j)) for s, e, i, j in zip(starts, ends, a, b, strict=True)]


def count_image(ev: Events, i: int, j: int) -> np.ndarray:
    """RGB uint8: positive events red, negative blue, on a dark grey ground, scaled by the 99th percentile."""
    h, w = ev.height, ev.width
    pos = np.zeros(h * w, np.float32)
    neg = np.zeros(h * w, np.float32)
    idx = ev.y[i:j].astype(np.int64) * w + ev.x[i:j].astype(np.int64)
    up = ev.p[i:j] > 0
    np.add.at(pos, idx[up], 1.0)
    np.add.at(neg, idx[~up], 1.0)
    nz = np.concatenate([pos[pos > 0], neg[neg > 0]])
    scale = float(np.percentile(nz, 99)) if len(nz) else 1.0
    a = np.clip(pos / max(scale, 1.0), 0, 1).reshape(h, w)
    b = np.clip(neg / max(scale, 1.0), 0, 1).reshape(h, w)
    img = np.full((h, w, 3), 24.0, np.float32)
    img[..., 0] += a * 231 + b * 40
    img[..., 1] += a * 70 + b * 110
    img[..., 2] += a * 40 + b * 231
    return np.clip(img, 0, 255).astype(np.uint8)


def time_surface(ev: Events, i: int, j: int, t_end: int, tau_ns: float) -> np.ndarray:
    """RGB uint8: exp(-(t_end - last event time) / tau) per pixel, positive in red, negative in blue."""
    h, w = ev.height, ev.width
    last_pos = np.full(h * w, -np.inf)
    last_neg = np.full(h * w, -np.inf)
    idx = ev.y[i:j].astype(np.int64) * w + ev.x[i:j].astype(np.int64)
    tt = ev.t[i:j].astype(np.float64)
    up = ev.p[i:j] > 0
    np.maximum.at(last_pos, idx[up], tt[up])
    np.maximum.at(last_neg, idx[~up], tt[~up])
    sp = np.exp(-(float(t_end) - last_pos) / tau_ns).reshape(h, w)
    sn = np.exp(-(float(t_end) - last_neg) / tau_ns).reshape(h, w)
    img = np.full((h, w, 3), 24.0, np.float32)
    img[..., 0] += sp * 231
    img[..., 1] += np.maximum(sp, sn) * 90
    img[..., 2] += sn * 231
    return np.clip(img, 0, 255).astype(np.uint8)


def accumulate(ev: Events, *, mode: str = "count", window_ms: float = 33.0, tau_ms: float = 30.0,
               max_frames: int = 600, min_events: int = 1) -> list[dict]:
    """One image per window with at least `min_events`: [{ts_ns (window end), n_events, image}]."""
    if mode not in ("count", "time_surface"):
        raise ValueError("mode must be count or time_surface")
    win = max(1, int(window_ms * 1e6))
    out = []
    for s, e, i, j in window_bounds(ev.t, int(ev.t[0]), int(ev.t[-1]), win):
        if j - i < min_events:
            continue
        img = count_image(ev, i, j) if mode == "count" else time_surface(ev, i, j, e, tau_ms * 1e6)
        out.append({"ts_ns": e, "start_ns": s, "n_events": j - i, "image": img})
        if len(out) >= max_frames:
            break
    return out


# --- import and frames, with the database -----------------------------------------------------------------

async def import_events(params: dict, *, user_id=None, progress: ProgressFn | None = None) -> dict:
    """An event file into an `event_stream` row (and a new session when none is named).

    params: source_uri, format (npy | h5 | mcap, default from the extension), sensor (the cam_id its frames
    get, default event_front), width / height (needed when the file does not say), columns and time_unit
    (.npy), topics (MCAP), time_offset_ns (added to every timestamp, to put the stream on a session's clock),
    session_id or new_session {vehicle_id, city}."""
    from core.storage import get_object_store
    from db.models import EventStream
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.imports.run import _acquire_source

    workdir = Path(tempfile.mkdtemp(prefix="events_import_"))
    try:
        local = await asyncio.to_thread(_acquire_source, str(params["source_uri"]), workdir)
        if local.is_dir():
            files = sorted(p for p in local.rglob("*") if p.suffix.lower() in (".npy", ".h5", ".hdf5", ".mcap"))
            if not files:
                raise EventFormatError("no .npy, .h5 or .mcap file in the upload")
            local = files[0]
        if progress is not None:
            await progress(0.2, "reading events")
        ev = await asyncio.to_thread(read_events, local, params.get("format"), columns=params.get("columns"),
                                     time_unit=params.get("time_unit"), width=params.get("width"),
                                     height=params.get("height"), topics=params.get("topics"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if ev.n > MAX_EVENTS:
        raise EventFormatError(f"{ev.n:,} events is more than one stream may hold ({MAX_EVENTS:,}); split the file")
    off = int(params.get("time_offset_ns") or 0)
    if off:
        ev.t = ev.t + off
    sensor = str(params.get("sensor") or "event_front")[:32]
    maker = get_sessionmaker()
    async with maker() as db:
        if params.get("session_id"):
            sid = uuid.UUID(str(params["session_id"]))
            if await db.get(DbSession, sid) is None:
                raise LookupError("session not found")
        else:
            ns = params.get("new_session") or {}
            sid = uuid.uuid4()
            db.add(DbSession(session_id=sid, vehicle_id=str(ns.get("vehicle_id") or "EVENT-CAM")[:64],
                             start_ts_ns=int(ev.t[0]), end_ts_ns=int(ev.t[-1]), city=ns.get("city"),
                             route=f"events:{sensor}", sensors={sensor: {"type": "event_camera",
                                                                         "width": ev.width, "height": ev.height}},
                             ontology_version=get_ontology().version, origin=REAL))
            await db.flush()
        stream_id = uuid.uuid4()
        store = get_object_store()
        store.ensure_bucket()
        if progress is not None:
            await progress(0.6, "storing events")
        uri = await asyncio.to_thread(store.put_bytes, f"events/{sid}/{stream_id}.npz", ev.to_npz_bytes())
        on = int((ev.p > 0).sum())
        dur = max(1, int(ev.t[-1] - ev.t[0]))
        db.add(EventStream(stream_id=stream_id, session_id=sid, sensor=sensor,
                           source=(params.get("format") or Path(str(params["source_uri"])).suffix.lstrip(".") or "npy")[:24],
                           uri=uri, width=ev.width, height=ev.height, n_events=ev.n, t0_ns=int(ev.t[0]),
                           t1_ns=int(ev.t[-1]), created_by=user_id,
                           stats={"positive": on, "negative": ev.n - on,
                                  "events_per_s": round(ev.n / (dur / 1e9), 1), "duration_s": round(dur / 1e9, 4)}))
        await db.commit()
    log.info("events.imported", stream=str(stream_id), session=str(sid), events=ev.n)
    return {"stream_id": str(stream_id), "session_id": str(sid), "events": ev.n, "width": ev.width,
            "height": ev.height, "t0_ns": int(ev.t[0]), "t1_ns": int(ev.t[-1])}


async def accumulate_stream(stream_id: uuid.UUID, *, mode: str = "count", window_ms: float = 33.0,
                            tau_ms: float = 30.0, max_frames: int = 600, replace: bool = True,
                            progress: ProgressFn | None = None) -> dict:
    """Accumulate a stored stream into frames on its session. With `replace`, the stream's earlier frames are
    removed first, except any a person has put a label on: those stay, and are counted."""
    import cv2
    from sqlalchemy import delete, exists, func, select

    from core.storage import get_object_store
    from db.models import EventStream, Frame, Object
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    store = get_object_store()
    async with maker() as db:
        st = await db.get(EventStream, stream_id)
        if st is None:
            raise LookupError("event stream not found")
        sid, sensor, uri = st.session_id, st.sensor, st.uri
    ev = await asyncio.to_thread(lambda: Events.from_npz_bytes(store.get_bytes(uri)))
    tag = f"stream:{stream_id}"
    kept = removed = 0
    if replace:
        async with maker() as db:
            mine = Frame.tags.contains([tag])
            has_obj = exists().where(Object.frame_id == Frame.frame_id)
            kept = int((await db.execute(select(func.count()).select_from(Frame).where(
                Frame.session_id == sid, mine, has_obj))).scalar() or 0)
            gone = (await db.execute(select(Frame.frame_id, Frame.img_uri).where(
                Frame.session_id == sid, mine, ~has_obj))).all()
            if gone:
                await db.execute(delete(Frame).where(Frame.frame_id.in_([g[0] for g in gone])))
            await db.commit()
            removed = len(gone)
        for _fid, u in gone:
            await asyncio.to_thread(store.remove, u)
    frames = await asyncio.to_thread(accumulate, ev, mode=mode, window_ms=window_ms, tau_ms=tau_ms,
                                     max_frames=max_frames)
    rows = []
    for k, fr in enumerate(frames):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(fr["image"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            continue
        fid = uuid.uuid4()
        img_uri = await asyncio.to_thread(store.put_bytes, f"frames/{sid}/{sensor}/{fr['ts_ns']}_{mode}.jpg",
                                          enc.tobytes(), "image/jpeg")
        rows.append(Frame(frame_id=fid, session_id=sid, ts_ns=int(fr["ts_ns"]), cam_id=sensor, img_uri=img_uri,
                          width=ev.width, height=ev.height, quality=1.0, origin=REAL, selected=True,
                          tags=[EVENT_TAG, f"event:{mode}", tag]))
        if progress is not None and k % 50 == 0:
            await progress(0.2 + 0.7 * (k + 1) / max(1, len(frames)), f"frame {k + 1} of {len(frames)}")
    acc = {"mode": mode, "window_ms": window_ms, "tau_ms": tau_ms if mode == "time_surface" else None,
           "frames": len(rows), "kept_labelled": kept, "removed": removed,
           "events_per_frame": round(float(np.mean([f["n_events"] for f in frames])), 1) if frames else 0}
    async with maker() as db:
        db.add_all(rows)
        st = await db.get(EventStream, stream_id)
        st.accumulation = acc
        from db.models import Session as DbSession

        sess = await db.get(DbSession, sid)
        if sess is not None and rows:
            sess.start_ts_ns = min(int(sess.start_ts_ns), min(r.ts_ns for r in rows))
            sess.end_ts_ns = max(int(sess.end_ts_ns), max(r.ts_ns for r in rows))
        await db.commit()
    log.info("events.accumulated", stream=str(stream_id), **{k: v for k, v in acc.items() if v is not None})
    return {"stream_id": str(stream_id), "session_id": str(sid), **acc,
            "first_frame_id": str(rows[0].frame_id) if rows else None}
