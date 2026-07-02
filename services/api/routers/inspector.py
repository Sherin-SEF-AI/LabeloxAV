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
