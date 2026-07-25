"""Extract the inertial channels (IMU, GNSS, time-sync pulse) a session needs for the SANYX time-sync, IMU,
and GPS checks, from its CHRONYX MCAP. This is a capability gate: a session with no inertial topics (an
imported image dataset, which is most of the current corpus) raises, and run.py simply skips the inertial
checks rather than run them on fabricated data.

Message log-times are schema-independent and always available, so the time-sync and rate checks run on any
MCAP that carries inertial topics. Decoded payload values (accel, gyro, GNSS fix, HDOP) are pulled
best-effort across the known Foxglove/ROS 2 schemas; when a field is not present the corresponding value
array stays empty and the value-based check (check_imu, check_gps) is skipped, never fabricated."""

from __future__ import annotations

import io

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Session as DbSession
from db.models import SessionIndex

log = get_logger("sanyx.inertial")

_IMU_HINTS = ("imu", "/mti", "xsens", "inertial")
_GNSS_HINTS = ("gnss", "gps", "navsat", "/fix", "ublox", "neo")
_PPS_HINTS = ("pps", "timesync", "time_sync", "sync", "stm32", "trigger")


def _classify(topic: str) -> str | None:
    t = topic.lower()
    if any(h in t for h in _IMU_HINTS):
        return "imu"
    if any(h in t for h in _GNSS_HINTS):
        return "gnss"
    if any(h in t for h in _PPS_HINTS):
        return "pps"
    return None


def _get_chain(obj, *names):
    """Best-effort nested-attribute read: _get_chain(msg, 'linear_acceleration', 'x')."""
    cur = obj
    for n in names:
        cur = getattr(cur, n, None)
        if cur is None:
            return None
    return cur


async def extract_inertial(db, session_id) -> dict:
    sess = await db.get(DbSession, session_id)
    if sess is None or not sess.mcap_uri:
        raise ValueError("session has no MCAP")
    idx = await db.get(SessionIndex, session_id)
    topics = (idx.topics if idx else {}) or {}
    selected = {t: _classify(t) for t in topics if _classify(t)}
    if not selected:
        raise ValueError("no inertial topics in session index")

    from mcap.reader import make_reader

    from services.ingest.reader_mcap import _decoder_factories

    store = get_object_store()
    raw = store.get_bytes(sess.mcap_uri)

    out: dict = {"imu_ts": [], "gnss_ts": [], "pps_ts": [], "accel": [], "gyro": [],
                 "fix_types": [], "hdop": [], "rtk_flags": []}
    reader = make_reader(io.BytesIO(raw), decoder_factories=_decoder_factories())
    for schema, channel, message, decoded in reader.iter_decoded_messages(topics=list(selected)):
        kind = selected.get(channel.topic)
        ts = int(message.log_time)
        if kind == "imu":
            out["imu_ts"].append(ts)
            ax = _get_chain(decoded, "linear_acceleration", "x")
            gx = _get_chain(decoded, "angular_velocity", "x")
            # Append accel AND gyro on every IMU message (0-filled when a field is absent) so accel, gyro, and
            # imu_ts stay the same length and index-aligned. Conditionally appending each let a message missing
            # one field desync the two streams, so accel[i] and gyro[i] no longer referred to the same sample.
            out["accel"].append([ax or 0.0, _get_chain(decoded, "linear_acceleration", "y") or 0.0,
                                 _get_chain(decoded, "linear_acceleration", "z") or 0.0])
            out["gyro"].append([gx or 0.0, _get_chain(decoded, "angular_velocity", "y") or 0.0,
                                _get_chain(decoded, "angular_velocity", "z") or 0.0])
        elif kind == "gnss":
            out["gnss_ts"].append(ts)
            status = _get_chain(decoded, "status", "status")
            if status is not None:
                out["fix_types"].append(int(status) + 2)  # ROS NavSatStatus is -1..2; shift to a 0..4 fix scale
            cov = getattr(decoded, "position_covariance", None)
            if cov is not None and len(cov):
                out["hdop"].append(float(cov[0]) ** 0.5)
        elif kind == "pps":
            out["pps_ts"].append(ts)

    log.info("sanyx.inertial", session=str(session_id),
             imu=len(out["imu_ts"]), gnss=len(out["gnss_ts"]), pps=len(out["pps_ts"]),
             accel=len(out["accel"]))
    return out
