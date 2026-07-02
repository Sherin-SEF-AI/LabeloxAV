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
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Session as DbSession
from services.api.deps import current_user, db_session, require_role

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


@router.get("/inspector/sessions", dependencies=[Depends(require_role("annotator"))])
async def list_sessions(limit: int = 100, db: AsyncSession = Depends(db_session)) -> list[dict]:
    """MCAP sessions available to inspect, each with its latest health verdict for the session-list chip."""
    from db.models import SessionHealth

    rows = (await db.execute(
        select(DbSession).where(DbSession.mcap_uri.isnot(None)).order_by(DbSession.created_at.desc()).limit(limit))).scalars().all()
    out = []
    for s in rows:
        h = (await db.execute(select(SessionHealth.verdict).where(SessionHealth.session_id == s.session_id)
                              .order_by(SessionHealth.created_at.desc()).limit(1))).scalar_one_or_none()
        out.append({"session_id": str(s.session_id), "vehicle_id": s.vehicle_id, "city": s.city,
                    "start_ts_ns": s.start_ts_ns, "end_ts_ns": s.end_ts_ns, "verdict": h})
    return out


@router.get("/inspector/sessions/{session_id}/frame-at", dependencies=[Depends(require_role("annotator"))])
async def frame_at(session_id: str, ts_ns: int, db: AsyncSession = Depends(db_session)) -> dict:
    """The extracted frame nearest a ts_ns (the image fast path and the Inspector-to-workspace deep link).
    Returns the frame id + its proxied image URL, or nulls if the session has no extracted frames."""
    from sqlalchemy import func

    from db.models import Frame

    sid = _parse_sid(session_id)
    row = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns, Frame.cam_id).where(Frame.session_id == sid)
        .order_by(func.abs(Frame.ts_ns - ts_ns)).limit(1))).first()
    if row is None:
        return {"frame_id": None, "ts_ns": None, "image_url": None, "cam_id": None}
    fid, fts, cam = row
    return {"frame_id": str(fid), "ts_ns": int(fts), "image_url": f"/api/frames/{fid}/image", "cam_id": cam}


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


@router.get("/inspector/sessions/{session_id}/mcap-url", dependencies=[Depends(require_role("annotator"))])
async def mcap_url(session_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    """A presigned MCAP URL plus the index summary, so the browser panels can read the file directly over
    HTTP range requests (the native Inspector's read path). Returns the time range and topics from the index
    if it has been built, so a panel knows the clock bounds without scanning the whole file."""
    sess = await db.get(DbSession, _parse_sid(session_id))
    if sess is None:
        raise HTTPException(404, "session not found")
    if not sess.mcap_uri:
        raise HTTPException(409, "session has no MCAP")
    from db.models import SessionIndex

    cfg = get_settings().inspector
    url = get_object_store().presigned_get(sess.mcap_uri, expires=cfg.presign_expiry_s)
    idx = await db.get(SessionIndex, _parse_sid(session_id))
    return {"session_id": session_id, "url": url, "expires_s": cfg.presign_expiry_s,
            "vehicle_id": sess.vehicle_id,
            "time_range": (idx.time_range if idx else [sess.start_ts_ns, sess.end_ts_ns]),
            "topics": (idx.topics if idx else {}), "gaps": (idx.gaps if idx else {})}


class LayoutIn(BaseModel):
    name: str
    panels: list
    is_default: bool = False


@router.get("/inspector/layouts", dependencies=[Depends(require_role("annotator"))])
async def list_layouts(db: AsyncSession = Depends(db_session), user=Depends(current_user)) -> dict:
    """This user's saved Inspector layouts, plus the config default as a fallback."""
    from db.models import InspectorLayout

    q = select(InspectorLayout).order_by(InspectorLayout.created_at.desc())
    if user:
        q = q.where((InspectorLayout.user_id == user.user_id) | (InspectorLayout.user_id.is_(None)))
    rows = (await db.execute(q)).scalars().all()
    layouts = [{"layout_id": str(r.layout_id), "name": r.name, "panels": r.panels, "is_default": r.is_default}
               for r in rows]
    return {"layouts": layouts, "config_default": get_settings().inspector.default_layout}


@router.post("/inspector/layouts", dependencies=[Depends(require_role("annotator"))])
async def save_layout(body: LayoutIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)) -> dict:
    """Save (or update by name) a named panel layout for this user."""
    import uuid as _uuid

    from db.models import InspectorLayout

    uid = user.user_id if user else None
    existing = (await db.execute(select(InspectorLayout).where(
        InspectorLayout.name == body.name, InspectorLayout.user_id == uid).limit(1))).scalar_one_or_none()
    if body.is_default and uid is not None:
        for r in (await db.execute(select(InspectorLayout).where(InspectorLayout.user_id == uid))).scalars().all():
            r.is_default = False
    if existing is None:
        existing = InspectorLayout(layout_id=_uuid.uuid4(), user_id=uid, name=body.name)
        db.add(existing)
    existing.panels = body.panels
    existing.is_default = body.is_default
    await db.commit()
    return {"layout_id": str(existing.layout_id), "name": existing.name, "is_default": existing.is_default}


@router.delete("/inspector/layouts/{layout_id}", dependencies=[Depends(require_role("annotator"))])
async def delete_layout(layout_id: str, db: AsyncSession = Depends(db_session)) -> dict:
    from db.models import InspectorLayout

    row = await db.get(InspectorLayout, _parse_sid(layout_id))
    if row is None:
        raise HTTPException(404, "layout not found")
    await db.delete(row)
    await db.commit()
    return {"deleted": layout_id}
