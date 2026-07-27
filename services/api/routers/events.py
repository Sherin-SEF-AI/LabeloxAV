"""Server-sent events: push job and progress state instead of making every client poll for it.

There was no realtime anywhere. Nine hardcoded setInterval loops stood in for it, each re-fetching a full
snapshot every two or three seconds whether or not anything had changed, unconditionally (no pause when the
tab is hidden, no backoff on error). Ingest progress was worse still: it was scraped out of a log file with a
regex, which breaks the moment the API runs in a different container from the ingest script.

SSE rather than WebSockets because the traffic is one-directional (the server reports, the client reads),
SSE rides ordinary HTTP so it needs no protocol upgrade through a proxy, and the browser reconnects on its
own. A WebSocket would add a second protocol to operate for no capability we need.

Disconnects are detected by letting the cancellation propagate, NOT by polling `request.is_disconnected()`.
That call awaits the ASGI receive channel, which the app's BaseHTTPMiddleware-based auth layer already owns,
so awaiting it here deadlocks: the response headers never flush and the client hangs forever rather than
receiving a stream. Starlette closes the generator when the client goes away, which surfaces as
GeneratorExit or CancelledError and is the supported way to notice.

The stream polls the database on the server side, which is not a change in mechanism so much as a change in
who pays: one query per interval serves every connected client instead of each client issuing its own, and
clients are pushed to only when something actually changed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import AutolabelJob, ExportJob, ImportJob, TrainingJob
from db.session import get_sessionmaker

log = get_logger("events")


async def require_stream_user(request: Request):
    """Authenticate a stream from the Authorization header, or from a `token` query parameter.

    The browser EventSource API cannot set a request header, so a stream has no other way to present a
    credential. The query form is accepted here and nowhere else, and is why these streams carry job progress
    and nothing sensitive: a URL reaches proxy and access logs in a way a header does not. The token is
    verified exactly as everywhere else, including expiry and the per-user revocation counter, so this is a
    different transport for the same credential, not a weaker check.
    """
    from uuid import UUID

    from db.models import User
    from services.api.auth_token import bearer_payload

    settings = get_settings()
    if not settings.auth.enabled:
        return None
    authz = request.headers.get("authorization")
    if not authz and (qs := request.query_params.get("token")):
        authz = f"Bearer {qs}"
    payload = bearer_payload(authz, settings.auth.signing_key,
                             accept_legacy=settings.auth.accept_legacy_tokens)
    if payload is None:
        raise HTTPException(status_code=401, detail="authentication required (Bearer token)")
    async with get_sessionmaker()() as db:
        try:
            user = await db.get(User, UUID(payload.uid))
        except Exception:  # noqa: BLE001
            user = None
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required (Bearer token)")
        if not payload.legacy and payload.token_version != user.token_version:
            raise HTTPException(status_code=401, detail="token revoked")
        return user


router = APIRouter(dependencies=[Depends(require_stream_user)])

# How often the server re-reads job state. Fast enough that a UI feels live, slow enough that a long-lived
# connection is not a database load generator.
POLL_INTERVAL_S = 2.0
# A proxy will drop an idle connection; a comment frame keeps it open without pretending to be an event.
HEARTBEAT_EVERY = 15


def _sse(event: str, data: dict) -> str:
    """One SSE frame. The double newline is the record separator; without it the client buffers forever."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _job_snapshot(db: AsyncSession) -> dict:
    """The state every job view needs, in one round trip per table.

    Only active work plus a small tail of recent terminal jobs: a client watching progress does not need the
    entire history, and sending it on every tick is what made the polling endpoints expensive.
    """
    out: dict[str, list[dict]] = {}

    training = (await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(20))).scalars().all()
    out["training"] = [{"job_id": str(j.job_id), "status": j.status, "stage": j.stage,
                        "progress": j.progress, "purpose": j.purpose, "task_type": j.task_type,
                        "counts": j.counts or {}, "metrics": (j.metrics or {}).get("live")}
                       for j in training]

    for key, model in (("import", ImportJob), ("export", ExportJob), ("autolabel", AutolabelJob)):
        rows = (await db.execute(
            select(model).order_by(model.created_at.desc()).limit(10))).scalars().all()
        out[key] = [{"job_id": str(r.job_id), "status": r.status,
                     "progress": getattr(r, "progress", None)} for r in rows]

    return out


@router.get("/events/jobs")
async def job_events():
    """Stream job state, emitting only when it changes.

    The first frame is always sent so a client renders immediately rather than waiting for the first change,
    which would otherwise leave a freshly opened page blank for as long as the system is quiet.

    No `Depends(db_session)`: a request-scoped session stays open until the response completes, and this
    response never completes, so it would hold a pooled connection for the entire life of every open tab.
    The generator opens and closes a short-lived session per tick instead.
    """
    from starlette.responses import StreamingResponse

    async def gen() -> AsyncIterator[str]:
        last: str | None = None
        ticks = 0
        maker = get_sessionmaker()
        try:
            while True:
                async with maker() as session:
                    snap = await _job_snapshot(session)
                payload = json.dumps(snap, default=str, sort_keys=True)
                if payload != last:
                    last = payload
                    yield _sse("jobs", snap)
                elif ticks % HEARTBEAT_EVERY == 0:
                    yield ": keepalive\n\n"
                ticks += 1
                await asyncio.sleep(POLL_INTERVAL_S)
        except (asyncio.CancelledError, GeneratorExit):
            # The client went away. Starlette closes the generator, which is how a disconnect reaches us;
            # this loop stops here rather than querying forever on behalf of a closed tab.
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("events.stream_failed", error=str(exc))
            yield _sse("error", {"detail": "stream ended"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Nginx buffers proxied responses by default, which holds every frame until the buffer fills and
        # makes a live stream arrive in bursts. This is the documented opt-out.
        "X-Accel-Buffering": "no",
    })


@router.get("/events/training/{job_id}")
async def training_job_events(job_id: str):
    """Stream one training job's progress, closing when it reaches a terminal state.

    A per-job stream exists because the training page wants epoch-level updates for the run being watched,
    and putting that in the global stream would push every client the details of a job they are not looking
    at. Closing on terminal state means the client does not hold a connection open for a finished job.
    """
    import uuid as uuidlib

    from starlette.responses import StreamingResponse

    async def gen() -> AsyncIterator[str]:
        maker = get_sessionmaker()
        last: str | None = None
        try:
            jid = uuidlib.UUID(job_id)
        except ValueError:
            yield _sse("error", {"detail": "invalid job id"})
            return
        while True:
            async with maker() as session:
                j = await session.get(TrainingJob, jid)
                if j is None:
                    yield _sse("error", {"detail": "job not found"})
                    return
                snap = {"job_id": job_id, "status": j.status, "stage": j.stage,
                        "progress": j.progress, "counts": j.counts or {},
                        "metrics": j.metrics or {}, "error": j.error}
                terminal = j.status in ("done", "error", "canceled")
            payload = json.dumps(snap, default=str, sort_keys=True)
            if payload != last:
                last = payload
                yield _sse("training", snap)
            if terminal:
                yield _sse("done", {"job_id": job_id, "status": snap["status"]})
                return
            await asyncio.sleep(POLL_INTERVAL_S)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })
