"""Integration endpoints: outbound webhooks and registered storage sources."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.integrations import storage_sources as src_svc
from services.integrations import webhooks as wh_svc

router = APIRouter()


class WebhookIn(BaseModel):
    url: str
    events: list[str] = []
    project_id: str | None = None
    secret: str | None = None


class SourceIn(BaseModel):
    name: str
    provider: str
    bucket: str
    prefix: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    credential_profile: str | None = None
    project_id: str | None = None


@router.get("/integrations/events")
async def known_events():
    """The events a webhook can subscribe to, named after what happened rather than which function ran."""
    return {"events": sorted(wh_svc.EVENTS)}


@router.post("/integrations/webhooks")
async def create_webhook(payload: WebhookIn, _user=Depends(require_role("admin")),
                         db: AsyncSession = Depends(db_session)):
    """Admin-only: a webhook sends data outward, so creating one is a disclosure decision. The signing secret
    is returned once here, because the receiver needs it to verify deliveries."""
    try:
        return await wh_svc.create_webhook(db, **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/integrations/webhooks")
async def list_webhooks(project_id: str | None = None, _user=Depends(require_role("admin")),
                        db: AsyncSession = Depends(db_session)):
    return {"webhooks": await wh_svc.list_webhooks(db, project_id)}


@router.delete("/integrations/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str, _user=Depends(require_role("admin")),
                         db: AsyncSession = Depends(db_session)):
    return await wh_svc.delete_webhook(db, webhook_id)


@router.post("/integrations/sources")
async def register_source(payload: SourceIn, _user=Depends(require_role("reviewer")),
                          db: AsyncSession = Depends(db_session)):
    try:
        return await src_svc.register_source(db, **payload.model_dump())
    except src_svc.SourceError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/integrations/sources")
async def list_sources(project_id: str | None = None, db: AsyncSession = Depends(db_session)):
    return {"sources": await src_svc.list_sources(db, project_id)}


@router.get("/integrations/sources/{source_id}/preview")
async def preview_source(source_id: str, limit: int = 50, _user=Depends(require_role("reviewer")),
                         db: AsyncSession = Depends(db_session)):
    """A bounded key listing, so a human can confirm a source points where they think before importing."""
    try:
        return await src_svc.preview_source(db, source_id, limit)
    except src_svc.SourceError as exc:
        raise HTTPException(400, str(exc)) from None


@router.delete("/integrations/sources/{source_id}")
async def delete_source(source_id: str, _user=Depends(require_role("reviewer")),
                        db: AsyncSession = Depends(db_session)):
    return await src_svc.delete_source(db, source_id)
