"""MCAP index service (M-I.1).

For each session, read the MCAP and produce one session_index row: every topic with its schema, message
count, measured rate, first and last ts_ns, and gap windows (silences longer than a per-topic threshold).
The measured rate is the real one (the IMU topic shows its true rate, not the nominal), which is the raw
material for the health checks, the timeline, and the topic browser. Per the compute-placement decision this
is CPU work in ingestion; it reads message log_times without decoding payloads, so it is cheap.

Raw is immutable: the indexer only reads the MCAP and writes a derived index row, stamped with the indexer
version for reproducibility.
"""

from __future__ import annotations

import io
import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Session as DbSession
from db.models import SessionIndex
from db.session import get_sessionmaker

log = get_logger("inspector.indexer")


def build_index_from_bytes(mcap_bytes: bytes, *, gap_min_factor: float) -> dict:
    """Parse an MCAP and return {topics, time_range, gaps}. A gap is a silence longer than gap_min_factor
    times the topic's nominal period (1 / measured rate)."""
    from mcap.reader import make_reader

    reader = make_reader(io.BytesIO(mcap_bytes))
    times: dict[str, list[int]] = defaultdict(list)
    schema_of: dict[str, str] = {}
    # iter_messages without a decoder factory reads record headers only (log_time), not payloads.
    for schema, channel, message in reader.iter_messages():
        times[channel.topic].append(int(message.log_time))
        if channel.topic not in schema_of:
            schema_of[channel.topic] = schema.name if schema is not None else ""

    topics: dict[str, dict] = {}
    gaps: dict[str, list] = {}
    all_lo: int | None = None
    all_hi: int | None = None
    for topic, ts in times.items():
        ts.sort()
        n = len(ts)
        lo, hi = ts[0], ts[-1]
        span_s = (hi - lo) / 1e9
        rate = round((n - 1) / span_s, 4) if span_s > 0 and n > 1 else 0.0
        topics[topic] = {"name": topic, "schema": schema_of.get(topic, ""), "count": n, "rate": rate,
                         "first_ts": lo, "last_ts": hi}
        if rate > 0:
            thresh = gap_min_factor * (1e9 / rate)
            wins = [[a, b] for a, b in zip(ts, ts[1:]) if (b - a) > thresh]
            if wins:
                gaps[topic] = wins
        all_lo = lo if all_lo is None else min(all_lo, lo)
        all_hi = hi if all_hi is None else max(all_hi, hi)

    return {"topics": topics, "time_range": [all_lo, all_hi] if all_lo is not None else None, "gaps": gaps}


async def index_session(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """Build (or rebuild) the index for one session and upsert the session_index row."""
    sess = await db.get(DbSession, session_id)
    if sess is None:
        raise ValueError("session not found")
    if not sess.mcap_uri:
        raise ValueError("session has no MCAP to index")
    cfg = get_settings().inspector
    data = get_object_store().get_bytes(sess.mcap_uri)
    idx = build_index_from_bytes(data, gap_min_factor=cfg.gap_min_factor)

    row = await db.get(SessionIndex, session_id)
    if row is None:
        row = SessionIndex(session_id=session_id)
        db.add(row)
    row.mcap_uri = sess.mcap_uri
    row.topics = idx["topics"]
    row.time_range = idx["time_range"]
    row.gaps = idx["gaps"]
    row.indexer_version = cfg.indexer_version
    await db.commit()
    log.info("inspector.index", session_id=str(session_id), topics=len(idx["topics"]),
             gaps=sum(len(v) for v in idx["gaps"].values()))
    return {"session_id": str(session_id), "topics": idx["topics"], "time_range": idx["time_range"],
            "gaps": idx["gaps"], "indexer_version": cfg.indexer_version}


async def index_session_bg(session_id: uuid.UUID) -> None:
    """Fire-and-forget index build for the ingestion hook (its own session)."""
    try:
        async with get_sessionmaker()() as db:
            await index_session(db, session_id)
    except Exception as exc:  # noqa: BLE001 - indexing never blocks ingestion
        log.error("inspector.index_failed", session_id=str(session_id), error=str(exc))


async def backfill(limit: int = 500) -> dict:
    """Index every MCAP session that has no current index. Returns counts."""
    cfg = get_settings().inspector
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(DbSession.session_id).outerjoin(SessionIndex, SessionIndex.session_id == DbSession.session_id)
            .where(DbSession.mcap_uri.isnot(None))
            .where((SessionIndex.session_id.is_(None)) | (SessionIndex.indexer_version != cfg.indexer_version))
            .limit(limit))).scalars().all()
    done, failed = 0, 0
    for sid in rows:
        try:
            async with maker() as db:
                await index_session(db, sid)
            done += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("inspector.backfill_skip", session_id=str(sid), error=str(exc))
            failed += 1
    log.info("inspector.backfill", indexed=done, failed=failed)
    return {"indexed": done, "failed": failed, "scanned": len(rows)}
