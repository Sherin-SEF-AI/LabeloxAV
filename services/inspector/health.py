"""Session health checks (M-I.2): the automated verdict that answers "did my recording work" in one glance,
before a single panel is opened.

Over the MCAP index (M-I.1) plus a light read of the file, it runs: rate-deviation per topic against the
rig's expected rates (the check that catches an IMU running at 247Hz against a 200Hz target), gap and
dropout detection, missing-topic detection against the rig manifest, cross-sensor time-offset sanity
(camera vs IMU vs GNSS on the PPS base), a GNSS fix-quality summary, and file integrity (truncated MCAP,
missing summary). It writes a session_health row with a pass, warn, or fail verdict; a fail flags the session
and gates it from auto-labeling until a human reviews, exactly as calibration validation gates 3D work.

Raw is immutable: this reads the MCAP and the index and writes only a derived health verdict.
"""

from __future__ import annotations

import io
import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Session as DbSession
from db.models import SessionHealth, SessionIndex

log = get_logger("inspector.health")

_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def _check(name: str, status: str, detail: str, evidence: dict | None = None) -> dict:
    return {"name": name, "status": status, "detail": detail, "evidence": evidence or {}}


def _match_topics(topics: dict, needle: str) -> list[str]:
    return [t for t in topics if needle in t]


def evaluate(index: dict, gnss_summary: dict, *, cfg) -> list[dict]:
    """Pure evaluation over the index + a GNSS summary. Returns the list of checks."""
    topics: dict = index.get("topics") or {}
    gaps: dict = index.get("gaps") or {}
    time_range = index.get("time_range")
    checks: list[dict] = []

    # 1. file integrity: the index built, has a time range, and every topic carries messages.
    empty = [t for t, i in topics.items() if i.get("count", 0) == 0]
    if not topics or time_range is None:
        checks.append(_check("file_integrity", "fail", "no readable topics or time range (truncated MCAP?)"))
    elif empty:
        checks.append(_check("file_integrity", "fail", f"topics with zero messages: {', '.join(empty)}",
                             {"empty_topics": empty}))
    else:
        checks.append(_check("file_integrity", "pass", f"{len(topics)} topics, contiguous summary"))

    # 2. missing topics + 3. rate deviation, against the rig manifest.
    for exp in cfg.expected_topics:
        matches = _match_topics(topics, exp["match"])
        if not matches:
            status = "fail" if exp.get("required") else "warn"
            checks.append(_check("missing_topic", status, f"no topic matching '{exp['match']}'",
                                 {"expected": exp}))
            continue
        for t in matches:
            measured = float(topics[t].get("rate", 0.0))
            expected = float(exp["rate"])
            dev = abs(measured - expected) / expected if expected else 0.0
            if dev <= cfg.rate_tolerance:
                st = "pass"
            elif dev <= cfg.rate_warn_tolerance:
                st = "warn"
            else:
                st = "fail"
            checks.append(_check("rate_deviation", st,
                                 f"{t} at {measured}Hz vs expected {expected}Hz ({round(dev * 100, 1)}% off)",
                                 {"topic": t, "measured": measured, "expected": expected}))

    # 4. dropouts: any gap longer than dropout_factor * nominal period is a fail; a smaller gap is a warn.
    for t, wins in gaps.items():
        rate = float(topics.get(t, {}).get("rate", 0.0))
        nominal = (1e9 / rate) if rate > 0 else 0.0
        worst = max(wins, key=lambda w: w[1] - w[0])
        span_s = (worst[1] - worst[0]) / 1e9
        is_dropout = nominal > 0 and (worst[1] - worst[0]) > cfg.dropout_factor * nominal
        checks.append(_check("dropout", "fail" if is_dropout else "warn",
                             f"{t}: {len(wins)} gap(s), worst {round(span_s, 3)}s",
                             {"topic": t, "worst_window": worst, "n_gaps": len(wins)}))

    # 5. cross-sensor time offset on the PPS base: first/last ts skew across camera/imu/gnss.
    anchors = {}
    for key in ("camera", "imu", "gnss"):
        m = _match_topics(topics, key)
        if m:
            anchors[key] = (topics[m[0]]["first_ts"], topics[m[0]]["last_ts"])
    if len(anchors) >= 2:
        firsts = [v[0] for v in anchors.values()]
        lasts = [v[1] for v in anchors.values()]
        skew_ms = max((max(firsts) - min(firsts)), (max(lasts) - min(lasts))) / 1e6
        st = "pass" if skew_ms <= cfg.max_cross_sensor_offset_ms else "warn"
        checks.append(_check("cross_sensor_offset", st,
                             f"camera/imu/gnss start-end skew {round(skew_ms, 1)}ms",
                             {"skew_ms": round(skew_ms, 1), "anchors": list(anchors)}))

    # 6. GNSS fix quality summary.
    if gnss_summary.get("present"):
        valid = gnss_summary.get("valid", 0)
        total = gnss_summary.get("total", 0)
        frac = valid / total if total else 0.0
        st = "pass" if frac >= 0.9 else ("warn" if frac >= 0.5 else "fail")
        checks.append(_check("gnss_fix", st, f"{valid}/{total} fixes with a valid position",
                             {"valid": valid, "total": total, "fix_fraction": round(frac, 3)}))

    return checks


def verdict_of(checks: list[dict]) -> str:
    worst = max((_ORDER[c["status"]] for c in checks), default=0)
    return {0: "pass", 1: "warn", 2: "fail"}[worst]


def _gnss_summary_from_bytes(mcap_bytes: bytes, gnss_topics: list[str]) -> dict:
    """Sample the GNSS topic and count messages with a plausible non-zero position."""
    if not gnss_topics:
        return {"present": False}
    from mcap.reader import make_reader

    reader = make_reader(io.BytesIO(mcap_bytes))
    total = valid = 0
    for _schema, channel, message in reader.iter_messages(topics=gnss_topics):
        total += 1
        try:
            d = json.loads(message.data)
            lat, lon = float(d.get("latitude", 0.0)), float(d.get("longitude", 0.0))
            if abs(lat) > 1e-6 and abs(lon) > 1e-6 and -90 <= lat <= 90 and -180 <= lon <= 180:
                valid += 1
        except Exception:  # noqa: BLE001 - a non-JSON GNSS payload still counts toward presence
            valid += 1
    return {"present": total > 0, "total": total, "valid": valid}


async def check_health(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """Run the health checks for a session (indexing it first if needed), write the verdict, and gate on fail."""
    from services.inspector.indexer import index_session

    sess = await db.get(DbSession, session_id)
    if sess is None:
        raise ValueError("session not found")
    if not sess.mcap_uri:
        raise ValueError("session has no MCAP")

    row = await db.get(SessionIndex, session_id)
    if row is None:
        await index_session(db, session_id)
        row = await db.get(SessionIndex, session_id)
    index = {"topics": row.topics, "gaps": row.gaps, "time_range": row.time_range}

    cfg = get_settings().inspector
    gnss_topics = _match_topics(index["topics"], "gnss")
    gnss_summary = {"present": False}
    try:
        data = get_object_store().get_bytes(sess.mcap_uri)
        gnss_summary = _gnss_summary_from_bytes(data, gnss_topics)
    except Exception as exc:  # noqa: BLE001 - GNSS summary is a bonus; integrity already covers a bad file
        log.warning("health.gnss_read_failed", session_id=str(session_id), error=str(exc))

    checks = evaluate(index, gnss_summary, cfg=cfg)
    verdict = verdict_of(checks)

    health = SessionHealth(session_id=session_id, checks=checks, verdict=verdict,
                           indexer_version=cfg.indexer_version)
    db.add(health)
    await db.commit()
    log.info("inspector.health", session_id=str(session_id), verdict=verdict,
             fails=[c["name"] for c in checks if c["status"] == "fail"])
    return {"session_id": str(session_id), "verdict": verdict, "checks": checks,
            "gated": verdict == "fail"}


async def latest_verdict(db: AsyncSession, session_id: uuid.UUID) -> str | None:
    """The most recent health verdict for a session, or None if never checked. Used to gate auto-labeling."""
    row = (await db.execute(select(SessionHealth).where(SessionHealth.session_id == session_id)
                            .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()
    return row.verdict if row else None


async def is_gated(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """True if the session's latest health verdict is a fail (excluded from auto-labeling until reviewed)."""
    return (await latest_verdict(db, session_id)) == "fail"
