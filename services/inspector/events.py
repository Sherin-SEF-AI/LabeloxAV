"""Timeline events (M-I.5): the typed markers the Inspector timeline shows, each clickable to jump the clock.

CAN events (hard braking, regen spikes) are detected from the MCAP CAN channel; scenario candidates, quality
flags, and gold/canary frames come from the existing platform records, keyed to ts_ns through the frame. This
is one of the integrations an external viewer cannot have: the events live in the same time base as playback.
"""

from __future__ import annotations

import io
import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import ErrorCandidate, Frame, Object, ScenarioCandidate
from db.models import Session as DbSession

log = get_logger("inspector.events")


def _can_events(mcap_bytes: bytes, can_topics: list[str], *, brake_kmh: float = 12.0, window_s: float = 0.5) -> list[dict]:
    """Detect hard braking (a fast speed drop) and regen spikes from the decoded CAN channel."""
    from mcap.reader import make_reader

    if not can_topics:
        return []
    speed: list[tuple[int, float]] = []
    regen: list[tuple[int, float]] = []
    for _schema, channel, message in make_reader(io.BytesIO(mcap_bytes)).iter_messages(topics=can_topics):
        try:
            d = json.loads(message.data)
        except Exception:  # noqa: BLE001
            continue
        sig = str(d.get("signal", "")).lower()
        val = d.get("value")
        if not isinstance(val, (int, float)):
            continue
        if "speed" in sig or channel.topic.endswith("/speed"):
            speed.append((int(message.log_time), float(val)))
        if "regen" in sig or int(d.get("id", 0)) == 0x249:
            regen.append((int(message.log_time), float(val)))

    events: list[dict] = []
    win_ns = int(window_s * 1e9)
    last = 0
    for i, (ts, v) in enumerate(speed):
        j = i
        while j > 0 and ts - speed[j - 1][0] < win_ns:
            j -= 1
        drop = speed[j][1] - v
        if drop >= brake_kmh and ts - last > win_ns:
            events.append({"ts_ns": ts, "kind": "hard_brake", "label": "hard braking",
                           "detail": f"-{round(drop, 1)} km/h in {window_s}s"})
            last = ts
    last = 0
    for k in range(1, len(regen)):
        if regen[k][1] - regen[k - 1][1] >= 20.0 and regen[k][0] - last > win_ns:
            events.append({"ts_ns": regen[k][0], "kind": "regen_spike", "label": "regen spike", "detail": ""})
            last = regen[k][0]
    return events


async def session_events(db: AsyncSession, session_id: uuid.UUID) -> list[dict]:
    """All timeline markers for a session: CAN events from the MCAP + scenario / quality / gold from records."""
    sess = await db.get(DbSession, session_id)
    if sess is None:
        return []
    events: list[dict] = []

    if sess.mcap_uri:
        try:
            from db.models import SessionIndex

            idx = await db.get(SessionIndex, session_id)
            can_topics = [t for t in (idx.topics or {}) if "can" in t.lower()] if idx else []
            if not can_topics:  # no index yet: fall back to a conventional CAN topic name
                can_topics = ["/can/speed"]
            events += _can_events(get_object_store().get_bytes(sess.mcap_uri), can_topics)
        except Exception as exc:  # noqa: BLE001 - CAN events are best-effort
            log.warning("events.can_failed", session_id=str(session_id), error=str(exc))

    for sc, ts in (await db.execute(
        select(ScenarioCandidate, Frame.ts_ns).join(Frame, Frame.frame_id == ScenarioCandidate.frame_id)
        .where(ScenarioCandidate.session_id == session_id))).all():
        events.append({"ts_ns": int(ts), "kind": "scenario", "label": sc.kind, "detail": sc.tag or ""})

    for ec_kind, ts in (await db.execute(
        select(ErrorCandidate.kind, Frame.ts_ns)
        .join(Object, Object.object_id == ErrorCandidate.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == session_id, ErrorCandidate.status == "pending").limit(200))).all():
        events.append({"ts_ns": int(ts), "kind": "quality", "label": ec_kind, "detail": ""})

    for (ts,) in (await db.execute(
        select(Frame.ts_ns).where(Frame.session_id == session_id, Frame.selected.is_(True)).limit(200))).all():
        events.append({"ts_ns": int(ts), "kind": "gold", "label": "gold frame", "detail": ""})

    events.sort(key=lambda e: e["ts_ns"])
    return events


async def annotations_at(db: AsyncSession, session_id: uuid.UUID, ts_ns: int) -> dict:
    """The nearest extracted frame's objects for the image overlay: class, confidence, box, state, source."""
    from sqlalchemy import func

    from services.autolabel.ontology import get_ontology

    row = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns, Frame.width, Frame.height).where(Frame.session_id == session_id)
        .order_by(func.abs(Frame.ts_ns - ts_ns)).limit(1))).first()
    if row is None:
        return {"frame_id": None, "objects": []}
    fid, fts, w, h = row
    onto = get_ontology()
    objs = (await db.execute(select(Object).where(Object.frame_id == fid, Object.state != "rejected"))).scalars().all()

    def _name(cid):
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    return {"frame_id": str(fid), "ts_ns": int(fts), "width": w, "height": h,
            "objects": [{"object_id": str(o.object_id), "class_name": _name(o.class_id), "conf": round(float(o.conf or 0), 3),
                         "bbox": [float(x) for x in o.bbox], "state": o.state, "source": o.source} for o in objs]}
