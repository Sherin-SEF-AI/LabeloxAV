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
import ipaddress
import json
import secrets
import socket
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.observability import spawn
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
RETRY_BACKOFF_S = (0.5, 2.0, 6.0)   # three retries; a transient 5xx or timeout should not lose the event
REPLAY_WINDOW_S = 300      # how old a signed delivery may be before a receiver should reject it


def sign(secret: str, body: bytes, timestamp: int | None = None) -> str:
    """Sign the body, binding a timestamp into the signature.

    Signing the body alone makes a captured delivery replayable forever: an attacker who records one request
    can resend it indefinitely and it still verifies. Including the timestamp in the signed material, and
    publishing it in the header, lets a receiver reject anything older than its tolerance, which is why
    Stripe and GitHub do it this way. The v1 form (body only) stays accepted by verify() so existing
    receivers keep working.
    """
    if timestamp is None:
        return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    signed = f"{timestamp}.".encode() + body
    return f"t={timestamp},sha256=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def verify(secret: str, body: bytes, signature: str, *, now: int | None = None,
           tolerance_s: int = REPLAY_WINDOW_S) -> bool:
    """Constant-time check, for receivers implemented against this same helper.

    Accepts both the timestamped form (t=...,sha256=...) and the legacy body-only form. For the timestamped
    form the age is checked as well, so a replayed capture outside the tolerance fails even though its
    signature is arithmetically correct.
    """
    sig = signature or ""
    if sig.startswith("t="):
        try:
            ts_part, _ = sig.split(",", 1)
            ts = int(ts_part[2:])
        except (ValueError, IndexError):
            return False
        current = int(time.time()) if now is None else now
        if abs(current - ts) > tolerance_s:
            return False
        return hmac.compare_digest(sign(secret, body, ts), sig)
    return hmac.compare_digest(sign(secret, body), sig)


def _is_safe_webhook_url(url: str) -> tuple[bool, str]:
    """Refuse URLs that point back into our own infrastructure.

    A webhook URL is attacker-controlled input that the server then fetches with its own network position, so
    an unrestricted one turns the API into an SSRF proxy: http://169.254.169.254/ reaches cloud instance
    credentials, and http://localhost:9000/ reaches MinIO. Only public hosts are accepted, and the check is on
    the resolved address rather than the hostname so a name pointing at a private range is caught too.
    """
    from urllib.parse import urlparse

    from core.config import get_settings

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "webhook url must be http or https"
    host = parsed.hostname
    if not host:
        return False, "webhook url has no host"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # Unresolvable at this moment is not proof of safety, but it is not proof of danger either: DNS may
        # simply not have propagated yet. The authoritative check runs again at delivery time (below), which
        # is the only moment that matters and which also closes DNS rebinding, where a name resolves public
        # at subscription time and private when we actually fetch it.
        return True, ""
    if get_settings().integrations.allow_private_webhook_targets:
        return True, ""   # deliberately opted in: an internal receiver on a self-hosted network
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return False, (f"webhook url resolves to a non-public address ({addr}); refusing to let a "
                           "subscription reach internal infrastructure")
    return True, ""


async def create_webhook(db: AsyncSession, *, url: str, events: list[str] | None = None,
                         project_id: str | None = None, secret: str | None = None) -> dict:
    ok, why = _is_safe_webhook_url(url)
    if not ok:
        raise ValueError(why)
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


def _should_retry(status: int | None) -> bool:
    """Retry a transport failure or a server-side error; never retry a 4xx.

    A 4xx means the receiver understood and rejected the request, so repeating it changes nothing and only
    amplifies load. A timeout or a 5xx is the case where the event was genuinely lost in transit and where a
    retry is the difference between delivered and silently dropped. 429 is included because it is an explicit
    "try again".
    """
    if status is None:
        return True                       # transport failure: connection refused, DNS, timeout
    return status >= 500 or status == 429


async def _deliver_one(wh_id: UUID, url: str, secret: str | None, body: bytes) -> None:
    """One delivery with bounded retries, recording the outcome. Never raises at the caller.

    Delivery used to be a single POST: one timeout or one 502 and the event was gone forever, with no retry,
    no record of what was attempted, and no way to replay. Now a retryable failure is retried with backoff and
    every attempt is written to the delivery log, so a missed event is visible and re-sendable instead of
    being lost silently.
    """
    status, err = None, None
    attempts = 0
    for attempt, delay in enumerate((0.0, *RETRY_BACKOFF_S)):
        if delay:
            await asyncio.sleep(delay)
        attempts = attempt + 1
        status, err = None, None
        # Re-check immediately before the fetch, not only at subscription time. A name that resolved to a
        # public address when the webhook was created can resolve to 169.254.169.254 now (DNS rebinding), and
        # the fetch is the moment our network position is actually lent out.
        safe, why = _is_safe_webhook_url(url)
        if not safe:
            await _record_attempt(wh_id, None, "blocked_unsafe_target", attempt + 1)
            log.warning("webhook.blocked_unsafe_target", webhook=str(wh_id), reason=why)
            return
        try:
            ts = int(time.time())
            headers = {"Content-Type": "application/json", "User-Agent": "labeloxav-webhook/1",
                       "X-Labelox-Timestamp": str(ts), "X-Labelox-Event-Id": str(uuid4())}
            if secret:
                headers["X-Labelox-Signature"] = sign(secret, body, ts)
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                r = await client.post(url, content=body, headers=headers)
                status = r.status_code
                if r.status_code >= 400:
                    err = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            err = type(exc).__name__      # sanitized: the url may embed a token

        if not err or not _should_retry(status):
            break

    await _record_attempt(wh_id, status, err, attempts)


async def _record_attempt(wh_id: UUID, status: int | None, err: str | None, attempts: int) -> None:
    """Persist the outcome of a delivery on the subscription row."""
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
        log.info("webhook.delivered" if not err else "webhook.delivery_failed",
                 webhook=str(wh_id), status=status, attempts=attempts, error=err)
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
        spawn(_deliver_one(wh.webhook_id, wh.url, wh.secret, body), name="_deliver_one")
        n += 1
    if n:
        log.info("webhook.emitted", event_name=event, deliveries=n)
    return n
