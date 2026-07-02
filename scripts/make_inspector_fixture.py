"""Generate a real multi-topic MCAP recording and register it as a session, so the Session Inspector can be
exercised end to end where no on-vehicle MCAP is present in the dev store.

This is a genuine recording, not mock data: real JPEG camera frames, a moving GNSS track, IMU at 200Hz, and
a CAN speed channel, all UTC-nanosecond timestamped on one time base (log_time == ts_ns), the invariant the
Inspector is built on. Foxglove-named schemas (foxglove.CompressedImage, foxglove.LocationFix) render
natively in Lichtblick; the IMU and CAN channels carry real numeric fields for the plot panels and give the
indexer + health checks their per-topic rates and gaps. Pass --gap-topic to seed a dropout for the health
demo, and --imu-rate to seed a wrong sensor rate.

    .venv/bin/python -m scripts.make_inspector_fixture --seconds 6 --imu-rate 200
    .venv/bin/python -m scripts.make_inspector_fixture --gap-topic /imu --imu-rate 247   # health demo
"""

from __future__ import annotations

import base64
import io
import json
import uuid

import click
import cv2
import numpy as np
from mcap.writer import Writer

from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns

_IMAGE_SCHEMA = {"type": "object", "properties": {
    "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
    "frame_id": {"type": "string"}, "data": {"type": "string", "contentEncoding": "base64"},
    "format": {"type": "string"}}}
_GNSS_SCHEMA = {"type": "object", "properties": {
    "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
    "frame_id": {"type": "string"}, "latitude": {"type": "number"}, "longitude": {"type": "number"},
    "altitude": {"type": "number"}}}
_IMU_SCHEMA = {"type": "object", "properties": {
    "linear_acceleration": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
    "angular_velocity": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}}}
_CAN_SCHEMA = {"type": "object", "properties": {
    "id": {"type": "integer"}, "signal": {"type": "string"}, "value": {"type": "number"}}}


def _ts_field(ts: int) -> dict:
    return {"sec": ts // 1_000_000_000, "nsec": ts % 1_000_000_000}


def build_mcap(seconds: float, imu_rate: float, gap_topic: str | None) -> tuple[bytes, int, int, dict]:
    """Write a real MCAP to memory. Returns (bytes, start_ts_ns, end_ts_ns, cam_frames{ts:jpeg}). gap_topic
    drops a 1s window; the returned camera JPEGs let the caller also register extracted frames."""
    buf = io.BytesIO()
    w = Writer(buf)
    w.start(profile="", library="labeloxav-inspector-fixture")
    rng = np.random.default_rng(7)
    start = now_ns()
    end = start + seconds_to_ns(seconds)
    gap_lo, gap_hi = start + seconds_to_ns(seconds * 0.4), start + seconds_to_ns(seconds * 0.4 + 1.0)

    def _schema(name: str, spec: dict) -> int:
        return w.register_schema(name=name, encoding="jsonschema", data=json.dumps(spec).encode())

    def _chan(topic: str, schema_id: int) -> int:
        return w.register_channel(topic=topic, message_encoding="json", schema_id=schema_id)

    def _in_gap(topic: str, ts: int) -> bool:
        return gap_topic == topic and gap_lo <= ts < gap_hi

    cam = _chan("/camera/cam_f", _schema("foxglove.CompressedImage", _IMAGE_SCHEMA))
    gnss = _chan("/gnss", _schema("foxglove.LocationFix", _GNSS_SCHEMA))
    imu = _chan("/imu", _schema("sensor.Imu", _IMU_SCHEMA))
    can = _chan("/can/speed", _schema("can.Signal", _CAN_SCHEMA))

    def _emit(chan: int, topic: str, ts: int, msg: dict) -> None:
        if _in_gap(topic, ts):
            return
        w.add_message(channel_id=chan, log_time=ts, publish_time=ts, data=json.dumps(msg).encode())

    # camera + GNSS at 10Hz
    cam_frames: dict[int, bytes] = {}
    n_cam = int(seconds * 10)
    for i in range(n_cam):
        ts = start + seconds_to_ns(i / 10.0)
        img = rng.integers(40, 220, size=(240, 320, 3), dtype=np.uint8)
        ok, jpg = cv2.imencode(".jpg", img)
        if ok and not _in_gap("/camera/cam_f", ts):
            cam_frames[ts] = jpg.tobytes()
            _emit(cam, "/camera/cam_f", ts, {"timestamp": _ts_field(ts), "frame_id": "cam_f",
                                             "format": "jpeg", "data": base64.b64encode(jpg.tobytes()).decode()})
        _emit(gnss, "/gnss", ts, {"timestamp": _ts_field(ts), "frame_id": "gnss",
                                  "latitude": 12.9716 + i * 5e-5, "longitude": 77.5946 + i * 5e-5, "altitude": 920.0})

    # IMU at imu_rate Hz (default 200; pass 247 to seed a wrong-rate health failure)
    n_imu = int(seconds * imu_rate)
    for i in range(n_imu):
        ts = start + int(i / imu_rate * 1e9)
        _emit(imu, "/imu", ts, {"linear_acceleration": {"x": float(0.2 * np.sin(i / 30)), "y": 0.0, "z": 9.81},
                                "angular_velocity": {"x": 0.0, "y": 0.0, "z": float(0.05 * np.cos(i / 25))}})

    # CAN speed at 100Hz: a drive-away ramp to 40, a hold, then a hard brake to 0 (a detectable event).
    n_can = int(seconds * 100)
    brake_i = int(n_can * 0.78)
    for i in range(n_can):
        ts = start + seconds_to_ns(i / 100.0)
        if i < n_can * 0.66:
            v = min(40.0, i * 0.1)
        elif i < brake_i:
            v = 40.0
        else:
            v = max(0.0, 40.0 - (i - brake_i) * 1.3)   # ~40 km/h shed in ~0.3s: a hard brake
        _emit(can, "/can/speed", ts, {"id": 0x247, "signal": "speed_kmh", "value": float(round(v, 2))})

    w.finish()
    return buf.getvalue(), start, end, cam_frames


async def _insert_frames(sid, uris_by_ts, gnss_track, vehicle, class_names):
    """Extract Frame rows + a couple of annotation boxes per frame + one scenario candidate, so the Inspector
    integrations (overlays, deep links, markers) have real data. Mirrors what ingestion + autolabel produce."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion, ScenarioCandidate
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    maker = get_sessionmaker()
    cids = [next(c.id for c in onto.classes if c.name == n) for n in class_names]
    frame_ids = []
    async with maker() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}))
            await db.flush()
        for k, (ts, uri) in enumerate(sorted(uris_by_ts.items())):
            fid = uuid.uuid4()
            frame_ids.append((ts, fid))
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri=uri, width=320, height=240,
                         quality=0.9, scene={"weather": "clear", "time_of_day": "day"}, selected=(k % 20 == 0)))
            await db.flush()
            # two boxes that drift across the frame, so overlays visibly track in motion
            for j, cid in enumerate(cids):
                x = 40 + (k * 3 + j * 120) % 220
                db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid, bbox=[float(x), 90.0, float(x + 70), 200.0],
                              conf=round(0.8 + 0.02 * j, 3), source="auto_accept", state="auto_accept", attrs={},
                              provenance={}, version=1))
        # a scenario candidate on a mid-clip frame (a timeline marker)
        if frame_ids:
            mid = frame_ids[len(frame_ids) // 2]
            db.add(ScenarioCandidate(session_id=sid, frame_id=mid[1], kind="high_risk", score=0.82,
                                     rare_classes=[], state="pending", tag="close interaction"))
        await db.commit()
    return len(frame_ids)


@click.command()
@click.option("--seconds", default=6.0, type=float)
@click.option("--imu-rate", default=200.0, type=float, help="IMU Hz; 247 seeds a wrong-rate health failure")
@click.option("--gap-topic", default=None, help="seed a 1s dropout on this topic for the health demo")
@click.option("--vehicle", default="TIGOR-07")
@click.option("--city", default="BLR")
@click.option("--with-frames", is_flag=True, help="also register extracted frames + annotations + a scenario")
def main(seconds: float, imu_rate: float, gap_topic: str | None, vehicle: str, city: str, with_frames: bool) -> None:
    import asyncio

    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    data, start, end, cam_frames = build_mcap(seconds, imu_rate, gap_topic)
    store = get_object_store()
    store.ensure_bucket()
    sid = uuid.uuid4()
    uri = store.put_bytes(f"raw/inspector-fixtures/{sid}.mcap", data, "application/octet-stream")

    from services.autolabel.ontology import get_ontology

    uris_by_ts = {}
    if with_frames:
        uris_by_ts = {ts: store.put_bytes(f"frames/{sid}/cam_f/{ts}.jpg", jpg, "image/jpeg") for ts, jpg in cam_frames.items()}

    async def _run() -> int:
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=start, end_ts_ns=end, city=city,
                             sensors={"cameras": ["cam_f"], "imu": True, "gnss": True, "can": True},
                             raw_uri=uri, mcap_uri=uri, ontology_version=get_ontology().version))
            await db.commit()
        return await _insert_frames(sid, uris_by_ts, None, vehicle, ["sedan", "pedestrian"]) if with_frames else 0

    n_frames = asyncio.run(_run())
    click.echo(f"session {sid}  vehicle {vehicle}  mcap {uri}  ({len(data)} bytes, {seconds}s, imu {imu_rate}Hz"
               + (f", gap on {gap_topic}" if gap_topic else "") + (f", {n_frames} frames+annotations" if with_frames else "") + ")")


if __name__ == "__main__":
    main()
