"""Position and speed read from the text a dashcam burns into its own video.

Fleet dashcams in India commonly stamp each frame with date, time, speed and GNSS fix, along the bottom:

    2026/05/31 15:23:18 080KM/H N:13.0578 E:77.4702 F7N TOUCH

That stamp was the only position this corpus had: Frame.gnss was set on 3 frames of 41,752, so ego poses,
ego-compensated propagation and anything geographic refused on every dashcam drive. This reads the stamp
back into Frame.gnss and Frame.ego_speed (m/s), after which services/intelligence/ego_pose.py builds a
measured trajectory from it like any other GNSS.

Reading is recognition-only OCR (rapidocr, CPU, about 0.1 s a frame) on the fixed strip where the stamp
sits; a session is probed on a few frames first, so footage without a stamp costs almost nothing. What is
read is checked before it is kept:
  - it must parse completely (time, speed, both coordinates with hemisphere letters);
  - coordinates must be on Earth and speed under 250 km/h;
  - a fix that would mean moving faster than the stamped speed allows since the last kept fix is dropped,
    because a misread digit is far more likely than a teleport.
Existing sensor GNSS is never overwritten.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from geoalchemy2.elements import WKTElement
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame

log = get_logger("overlay_telemetry")

# Where the stamp sits, as fractions of the frame: the bottom-left band.
STRIP = (0.935, 0.975, 0.0, 0.42)  # y0, y1, x0, x1
PROBE_FRAMES = 6
MAX_SPEED_KMH = 250.0
# Slack on the speed check, in metres: the fix is rounded to 4 decimals (about 11 m) and the stamp's speed
# is the speed at one instant.
JUMP_SLACK_M = 60.0

_PAT = re.compile(
    r"(\d{4})[/-](\d{2})[/-](\d{2})\s*(\d{2}):(\d{2}):(\d{2})\D*?(\d{1,3})\s*KM/?H"
    r".*?([NS])\s*[:.]?\s*(\d{1,2}[.,]\d{3,6}).*?([EW])\s*[:.]?\s*(\d{1,3}[.,]\d{3,6})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Telemetry:
    when: datetime
    speed_kmh: float
    lat: float
    lon: float


def _normalise(text: str) -> str:
    """Undo the OCR confusions this font produces inside numbers: '3L' for 31, 'T5' for 15, 'O' for 0.

    Only characters touching a digit are changed, so the field labels (KM/H, N:, E:) and hemisphere letters
    are left alone.
    """
    t = re.sub(r"(?<=\d)[LlI|]|[LlI|T](?=\d)", "1", text)
    return re.sub(r"(?<=\d)O|O(?=\d)", "0", t)


def parse_stamp(text: str) -> Telemetry | None:
    m = _PAT.search(_normalise(text or ""))
    if not m:
        return None
    y, mo, d, hh, mi, ss, kmh, ns, lat, ew, lon = m.groups()
    try:
        when = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss))
    except ValueError:
        return None
    la = float(lat.replace(",", ".")) * (-1 if ns.upper() == "S" else 1)
    lo = float(lon.replace(",", ".")) * (-1 if ew.upper() == "W" else 1)
    speed = float(kmh)
    if not (-90 <= la <= 90 and -180 <= lo <= 180) or speed > MAX_SPEED_KMH or (la == 0 and lo == 0):
        return None
    return Telemetry(when, speed, la, lo)


@lru_cache(maxsize=1)
def _ocr() -> Any:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def available() -> bool:
    return _ocr() is not None


def read_overlay(image_bgr: np.ndarray | None) -> Telemetry | None:
    """The stamp on one frame, or None when there is none, it is unreadable, or OCR is not installed."""
    ocr = _ocr()
    if ocr is None or image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]
    y0, y1, x0, x1 = STRIP
    strip = image_bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if strip.size == 0:
        return None
    res, _ = ocr(strip, use_det=False, use_cls=False, use_rec=True)
    return parse_stamp(" ".join(str(r[0]) for r in (res or [])))


def load_image(store, uri: str) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 - one unreadable image is a frame without a stamp
        return None


def _metres(a: Telemetry, b: Telemetry) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


async def backfill_session(db: AsyncSession, store, session_id: uuid.UUID) -> dict:
    """Read every frame's stamp in one session and write position and speed where there were none."""
    if not available():
        return {"session_id": str(session_id), "read": 0, "written": 0,
                "reason": "rapidocr is not installed, so the stamp cannot be read"}
    frames = (await db.execute(
        select(Frame.frame_id, Frame.img_uri, Frame.gnss.is_not(None), Frame.cam_id, Frame.ts_ns)
        .where(Frame.session_id == session_id)
        .order_by(Frame.cam_id, Frame.ts_ns))).all()
    if not frames:
        return {"session_id": str(session_id), "read": 0, "written": 0, "reason": "no frames"}

    def _load(uri: str) -> np.ndarray | None:
        return load_image(store, uri)

    step = max(1, len(frames) // PROBE_FRAMES)
    if not any(read_overlay(_load(uri)) for _fid, uri, _g, _c, _t in frames[::step][:PROBE_FRAMES]):
        return {"session_id": str(session_id), "read": 0, "written": 0,
                "reason": "no telemetry stamp found on the probed frames"}

    read = written = rejected = 0
    by_cam: dict[str, list] = {}
    for row in frames:
        by_cam.setdefault(row[3], []).append(row)
    for cam_frames in by_cam.values():
        # Pass 1: read every stamp, keeping the plausible ones. The receiver updates every few seconds and
        # holds the fix in between, so a real update arrives as one jump after several unchanged frames, and
        # the time it covered is since the position last changed, not since the previous frame.
        kept: dict[Any, Telemetry] = {}
        last: Telemetry | None = None
        since: datetime | None = None
        for fid, uri, _has_gnss, _cam, _ts in cam_frames:
            t = read_overlay(_load(uri))
            if t is None:
                continue
            read += 1
            if last is not None and since is not None:
                moved = _metres(last, t)
                dt = abs((t.when - since).total_seconds())
                allowed = max(t.speed_kmh, last.speed_kmh) / 3.6 * max(dt, 1.0) * 2.0 + JUMP_SLACK_M
                if moved > allowed:
                    rejected += 1
                    continue
                if moved > 0:
                    since = t.when
            else:
                since = t.when
            last = t
            kept[fid] = t

        # Pass 2: where the fix changed. Each change is placed at the frame that first showed it, and every
        # frame between two changes is placed on the line between them by its video timestamp. Held fixes
        # would make the vehicle stand still for seconds and then jump, which is what an ego transform
        # between neighbouring frames reads as "no motion".
        changes: list[tuple[int, float, float]] = []
        for fid, _uri, _g, _c, ts in cam_frames:
            t = kept.get(fid)
            if t is not None and (not changes or (t.lat, t.lon) != (changes[-1][1], changes[-1][2])):
                changes.append((int(ts), t.lat, t.lon))
        if not changes:
            continue
        j = 0
        for fid, _uri, has_gnss, _c, ts in cam_frames:
            ts = int(ts)
            t = kept.get(fid)
            if ts < changes[0][0] and t is None:
                continue
            while j + 1 < len(changes) and changes[j + 1][0] <= ts:
                j += 1
            if j + 1 < len(changes) and ts >= changes[j][0]:
                (t0, la0, lo0), (t1, la1, lo1) = changes[j], changes[j + 1]
                f = (ts - t0) / max(1, t1 - t0)
                lat, lon = la0 + (la1 - la0) * f, lo0 + (lo1 - lo0) * f
            elif t is not None:
                lat, lon = t.lat, t.lon
            else:
                continue
            values: dict[str, Any] = {}
            if t is not None:
                values["ego_speed"] = t.speed_kmh / 3.6
            if not has_gnss:
                values["gnss"] = WKTElement(f"POINT({lon} {lat})", srid=4326)
            if values:
                await db.execute(update(Frame).where(Frame.frame_id == fid).values(**values))
                written += 1
    await db.commit()
    log.info("overlay.backfilled", session=str(session_id), frames=len(frames), read=read, written=written,
             rejected=rejected)
    return {"session_id": str(session_id), "frames": len(frames), "read": read, "written": written,
            "rejected_jumps": rejected}


__all__ = ["Telemetry", "available", "backfill_session", "parse_stamp", "read_overlay"]
