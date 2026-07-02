"""Session Inspector API.

M-I.0 (the day-one escape hatch): presign a session's MCAP from the object store and build a Lichtblick
remote-file deep link, so any session can be opened in the self-hosted Lichtblick (open-source Foxglove
fork) for full-power inspection while the native Inspector is built. Later milestones add the index, health,
and panel-serving endpoints on this same router.
"""

from __future__ import annotations

import uuid
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Session as DbSession
from services.api.deps import db_session, require_role

log = get_logger("api_inspector")
router = APIRouter()


def _lichtblick_url(mcap_url: str) -> str:
    """Build a Lichtblick remote-file deep link. Lichtblick's URL scheme (inherited from Foxglove) selects a
    data source with ds=<id> and passes its arguments with the ds. prefix, so a remote MCAP is
    ds=remote-file&ds.url=<file>. The base url and source id are config-driven."""
    cfg = get_settings().inspector
    base = cfg.lichtblick_base_url.rstrip("/")
    return f"{base}/?ds={quote(cfg.lichtblick_ds_id)}&ds.url={quote(mcap_url, safe='')}"


@router.get("/inspector/sessions/{session_id}/lichtblick", dependencies=[Depends(require_role("annotator"))])
async def lichtblick_link(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """Presign the session's MCAP and return a Lichtblick deep link that loads it as a remote file."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(400, "invalid session id") from exc
    sess = await db.get(DbSession, sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    if not sess.mcap_uri:
        raise HTTPException(409, "session has no MCAP recording to inspect")
    cfg = get_settings().inspector
    mcap_url = get_object_store().presigned_get(sess.mcap_uri, expires=cfg.presign_expiry_s)
    deep = _lichtblick_url(mcap_url)
    log.info("inspector.lichtblick_link", session_id=session_id)
    return {"session_id": session_id, "url": deep, "mcap_url": mcap_url, "expires_s": cfg.presign_expiry_s}


def _parse_sid(session_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(400, "invalid session id") from exc


@router.post("/inspector/sessions/{session_id}/index", dependencies=[Depends(require_role("reviewer"))])
async def build_index(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """Build (or rebuild) the MCAP index for a session: per-topic schema, count, measured rate, time range,
    and gap windows. Cheap; reads message log_times without decoding payloads."""
    from services.inspector.indexer import index_session

    try:
        return await index_session(db, _parse_sid(session_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/inspector/sessions/{session_id}/index", dependencies=[Depends(require_role("annotator"))])
async def get_index(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """The stored MCAP index for a session (topics, time range, gaps)."""
    from db.models import SessionIndex

    row = await db.get(SessionIndex, _parse_sid(session_id))
    if row is None:
        raise HTTPException(404, "session not indexed yet")
    return {"session_id": session_id, "mcap_uri": row.mcap_uri, "topics": row.topics,
            "time_range": row.time_range, "gaps": row.gaps, "indexer_version": row.indexer_version,
            "built_at": row.built_at.isoformat() if row.built_at else None}


@router.post("/inspector/index/backfill", dependencies=[Depends(require_role("reviewer"))])
async def index_backfill(limit: int = 500, db: AsyncSession = Depends(db_session)) -> dict:
    """Index every MCAP session that is missing a current index."""
    from services.inspector.indexer import backfill

    return await backfill(limit=limit)


@router.post("/inspector/sessions/{session_id}/health", dependencies=[Depends(require_role("reviewer"))])
async def run_health(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """Run the session health checks (rate, gaps, missing topics, cross-sensor offset, GNSS, integrity) and
    record a pass/warn/fail verdict. A fail gates the session from auto-labeling until reviewed."""
    from services.inspector.health import check_health

    try:
        return await check_health(db, _parse_sid(session_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/inspector/sessions/{session_id}/health", dependencies=[Depends(require_role("annotator"))])
async def get_health(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """The latest health verdict + per-check detail for a session."""
    from db.models import SessionHealth

    sid = _parse_sid(session_id)
    row = (await db.execute(select(SessionHealth).where(SessionHealth.session_id == sid)
                            .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()
    if row is None:
        return {"session_id": session_id, "verdict": None, "checks": []}
    return {"session_id": session_id, "verdict": row.verdict, "checks": row.checks,
            "created_at": row.created_at.isoformat() if row.created_at else None}


@router.post("/inspector/health/sweep", dependencies=[Depends(require_role("reviewer"))])
async def health_sweep(limit: int = 500, db: AsyncSession = Depends(db_session)) -> dict:
    """Run health checks across all MCAP sessions (indexing as needed). Returns the verdict tally."""
    from collections import Counter

    from services.inspector.health import check_health

    sids = (await db.execute(select(DbSession.session_id).where(DbSession.mcap_uri.isnot(None)).limit(limit))).scalars().all()
    tally: Counter = Counter()
    for sid in sids:
        try:
            r = await check_health(db, sid)
            tally[r["verdict"]] += 1
        except Exception:  # noqa: BLE001
            tally["error"] += 1
    return {"checked": len(sids), "verdicts": dict(tally)}
