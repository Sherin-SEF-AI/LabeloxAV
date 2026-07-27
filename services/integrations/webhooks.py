"""Outbound webhooks: let an external pipeline react to what happens here instead of polling.

Every delivery carries an HMAC-SHA256 signature over the exact body bytes, in the header
`X-Labelox-Signature: sha256=<hex>`, the same scheme GitHub and Stripe use. A receiver that does not verify it
cannot tell a real delivery from anyone who learned the URL, so a webhook without a signature is an
unauthenticated write into whatever it triggers.

Delivery is best-effort and bounded: a slow or dead endpoint must never be able to stall the request that
emitted the event, so dispatch runs detached with a short timeout and failures are recorded on the
subscription rather than raised at the caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Webhook
from db.session import get_sessionmaker

log = get_logger("webhooks")

# The events worth subscribing to. Named after what happened, not which function ran, so a consumer is not
# coupled to our internal call graph.
EVENTS = (
    "job.assigned", "job.submitted", "job.rejected",
    "issue.opened", "issue.resolved",
    "asset.labeled", "annotation.created",
    "export.completed", "model.promoted", "drift.breached",
    # Security-domain events (SEC-M8): a downstream consumer (e.g. Sentigon) subscribes to these to react to
    # what a static-camera deployment sees. Emitted from the security path (services/anpr/events.py); the
    # webhook mechanism itself is domain-neutral.
    "anpr.read", "anpr.watchlist_hit", "security.event",
)

TIMEOUT_S = 5.0
MAX_FAILURES = 20          # after this many consecutive failures the subscription deactivates itself


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify(secret: str, body: bytes, signature: str) -> bool:
    """Constant-time check, for receivers implemented against this same helper."""
    return hmac.compare_digest(sign(secret, body), signature or "")


async def create_webhook(db: AsyncSession, *, url: str, events: list[str] | None = None,
                         project_id: str | None = None, secret: str | None = None) -> dict:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("webhook url must be http or https")
    for e in events or []:
        if e not in EVENTS:
            raise ValueError(f"unknown event {e!r} (known: {sorted(EVENTS)})")
    wh = Webhook(url=url, events=list(events or []),
                 project_id=UUID(project_id) if project_id else None,
                 secret=secret or secrets.token_hex(24))
    db.add(wh)
    await db.commit()
    log.info("webhook.created", webhook=str(wh.webhook_id), url=url, events=wh.events)
    # The secret is returned once, at creation, because the receiver needs it to verify deliveries.
    return {**_dict(wh), "secret": wh.secret}


def _dict(w: Webhook) -> dict:
    return {"webhook_id": str(w.webhook_id), "url": w.url, "events": w.events or [],
            "project_id": str(w.project_id) if w.project_id else None, "active": w.active,
            "last_status": w.last_status, "last_error": w.last_error,
            "failure_count": w.failure_count,
            "last_delivery_at": w.last_delivery_at.isoformat() if w.last_delivery_at else None,
            "created_at": w.created_at.isoformat() if w.created_at else None}


async def list_webhooks(db: AsyncSession, project_id: str | None = None) -> list[dict]:
    stmt = select(Webhook)
    if project_id:
        stmt = stmt.where(Webhook.project_id == UUID(project_id))
    rows = (await db.execute(stmt.order_by(Webhook.created_at.desc()).limit(200))).scalars().all()
    return [_dict(w) for w in rows]


async def delete_webhook(db: AsyncSession, webhook_id: str) -> dict:
    w = await db.get(Webhook, UUID(webhook_id))
    if w is None:
        return {"deleted": False}
    await db.delete(w)
    await db.commit()
    return {"deleted": True}


async def _deliver_one(wh_id: UUID, url: str, secret: str | None, body: bytes) -> None:
    """One delivery, recording the outcome on the subscription. Never raises at the caller."""
    status, err = None, None
    try:
        headers = {"Content-Type": "application/json", "User-Agent": "labeloxav-webhook/1"}
        if secret:
            headers["X-Labelox-Signature"] = sign(secret, body)
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(url, content=body, headers=headers)
            status = r.status_code
            if r.status_code >= 400:
                err = f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        err = type(exc).__name__          # sanitized: the url may embed a token

    try:
        async with get_sessionmaker()() as db:
            wh = await db.get(Webhook, wh_id)
            if wh is None:
                return
            wh.last_status, wh.last_error = status, err
            wh.last_delivery_at = datetime.now(UTC)
            if err:
                wh.failure_count += 1
                # A permanently dead endpoint is switched off rather than retried forever, and the reason is
                # left on the row so it is visible instead of merely silent.
                if wh.failure_count >= MAX_FAILURES:
                    wh.active = False
                    log.warning("webhook.deactivated", webhook=str(wh_id), failures=wh.failure_count)
            else:
                wh.failure_count = 0
            await db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("webhook.status_write_failed", webhook=str(wh_id), error=str(exc))


async def emit(event: str, payload: dict, *, project_id: str | None = None) -> int:
    """Fan an event out to every matching active subscription. Returns how many deliveries were scheduled.

    Fire and forget by design: the caller is usually inside a request, and a slow receiver must not be able to
    hold that request open. Failures land on the subscription, not on the user.
    """
    if event not in EVENTS:
        log.warning("webhook.unknown_event", event_name=event)
        return 0
    try:
        async with get_sessionmaker()() as db:
            stmt = select(Webhook).where(Webhook.active.is_(True))
            rows = (await db.execute(stmt)).scalars().all()
    except Exception as exc:  # noqa: BLE001
        log.warning("webhook.lookup_failed", error=str(exc))
        return 0

    body = json.dumps({"event": event, "project_id": project_id,
                       "at": datetime.now(UTC).isoformat(), "data": payload},
                      default=str).encode()

    n = 0
    for wh in rows:
        if wh.events and event not in wh.events:
            continue
        if wh.project_id and project_id and str(wh.project_id) != str(project_id):
            continue
        asyncio.create_task(_deliver_one(wh.webhook_id, wh.url, wh.secret, body))
        n += 1
    if n:
        log.info("webhook.emitted", event_name=event, deliveries=n)
    return n
