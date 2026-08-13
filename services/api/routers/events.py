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
from sqlalchemy import func, select
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
# Held but not moving. `pending` covers work nobody has picked up; `queued-cloud` is parked for the A100 on
# purpose. Neither is progress, and both are the answer to why a quiet system is quiet.
WAITING_STATUSES = ("pending", "queued", "queued-cloud")
# Slower than the job stream: these are gauges, not events, and a console that redraws twice a second reads
# as noise rather than as information. Fast enough that a job starting is visible while you are looking.
SYSTEM_INTERVAL_S = 3.0


# ---------------------------------------------------------------- wakeups
#
# The streams above poll the database, which is the right default: one query per interval serves every
# connected client. But an event a user is waiting on (a notification, a finished job) should not sit for a
# whole interval, so an emitter can nudge the topic and the next tick happens immediately.
#
# In-process only, and deliberately so. Every stream still polls, so a nudge that never arrives (because it
# was raised in a different worker) costs latency and never a lost event. A cross-process bus here would
# make correctness depend on the bus being up, which is a worse failure than being a second late.

_wakes: dict[str, asyncio.Event] = {}


def _wake_event(topic: str) -> asyncio.Event:
    ev = _wakes.get(topic)
    if ev is None:
        ev = _wakes[topic] = asyncio.Event()
    return ev


def publish(topic: str, payload: dict | None = None) -> None:
    """Nudge a topic so its streams tick now instead of on their next interval. Never raises: an emitter
    must not fail because nobody is listening."""
    try:
        _wake_event(topic).set()
    except Exception:  # noqa: BLE001
        pass


async def _sleep_or_wake(topic: str, seconds: float) -> None:
    """Wait out the poll interval, but return early if someone publishes to this topic."""
    ev = _wake_event(topic)
    try:
        await asyncio.wait_for(ev.wait(), timeout=seconds)
    except TimeoutError:
        return
    finally:
        ev.clear()


def _sse(event: str, data: dict) -> str:
    """One SSE frame. The double newline is the record separator; without it the client buffers forever."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _job_snapshot(db: AsyncSession) -> dict:
    """The state every job view needs, in one round trip per table.

    Only active work plus a small tail of recent terminal jobs: a client watching progress does not need the
    entire history, and sending it on every tick is what made the polling endpoints expensive. The one thing
    that cannot be answered from a tail, how much work is queued in total, is carried alongside as a count.
    """
    out: dict[str, object] = {}

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
                     "progress": getattr(r, "progress", None),
                     "counts": getattr(r, "counts", None) or {},
                     "error": getattr(r, "error", None)} for r in rows]

    # How much work is held but not moving, counted over every row rather than over the window above.
    #
    # The lists are capped at a recent tail, which is right for showing progress and wrong for answering "why
    # is nothing happening". This deployment holds 67 autolabel jobs parked for a cloud A100 since late June;
    # all of them are older than the ten most recent, so a client counting the window reported one queued job
    # when there were sixty-eight. Under-reporting is worse than not reporting: it reads as a healthy system.
    waiting: dict[str, int] = {}
    for wkey, wmodel in (("training", TrainingJob), ("import", ImportJob),
                         ("export", ExportJob), ("autolabel", AutolabelJob)):
        waiting[wkey] = int((await db.execute(
            select(func.count()).select_from(wmodel)
            .where(wmodel.status.in_(WAITING_STATUSES)))).scalar() or 0)
    out["waiting"] = waiting

    # Ingest progress rides the same stream rather than getting one of its own. It is the signal the home
    # page shows while a fleet sweep runs, and it used to be scraped out of a log file with a regex, which
    # breaks the moment the API runs in a different container from the ingest script.
    try:
        from services.ingest.progress import read_progress, with_frame_count

        out["ingest"] = [await with_frame_count(db, await read_progress(db))]
    except Exception:  # noqa: BLE001 - an absent progress source must not take the whole stream down
        out["ingest"] = []

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
                await _sleep_or_wake("jobs", POLL_INTERVAL_S)
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


@router.get("/events/notifications")
async def notification_events(request: Request):
    """Stream this user's unread count and newest notifications, so the bell updates without a poll.

    Scoped to the caller. A notification stream that pushed everyone's events to everyone would leak who is
    being assigned what, which on a corpus with a review hierarchy is real information about people.
    """
    from starlette.responses import StreamingResponse

    from services import notify

    user = await require_stream_user(request)
    if user is None:
        # Auth disabled (unit tests): nothing to scope to, so there is nothing honest to stream.
        async def empty() -> AsyncIterator[str]:
            yield _sse("notifications", {"unread": 0, "notifications": []})

        return StreamingResponse(empty(), media_type="text/event-stream")

    async def gen() -> AsyncIterator[str]:
        last: str | None = None
        ticks = 0
        maker = get_sessionmaker()
        try:
            while True:
                async with maker() as session:
                    fresh = await session.get(type(user), user.user_id)
                    if fresh is None:
                        break
                    unread = await notify.unread_count(session, fresh)
                    listing = await notify.list_for(session, fresh, limit=10)
                snap = {"unread": int(unread), "notifications": listing["notifications"]}
                payload = json.dumps(snap, default=str, sort_keys=True)
                if payload != last:
                    last = payload
                    yield _sse("notifications", snap)
                elif ticks % HEARTBEAT_EVERY == 0:
                    yield ": keepalive\n\n"
                ticks += 1
                await _sleep_or_wake("notifications", POLL_INTERVAL_S * 2)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("events.notification_stream_failed", error=str(exc))
            yield _sse("error", {"detail": "stream ended"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })


@router.get("/events/system")
async def system_events():
    """Stream machine state for the console.

    Unlike the job stream this emits every tick rather than only on change: utilisation and temperature are
    never equal two ticks running, so change detection would emit constantly anyway, and a console whose
    numbers freeze is indistinguishable from one whose connection died.
    """
    from starlette.responses import StreamingResponse

    from services.hardening.resources import snapshot

    async def gen() -> AsyncIterator[str]:
        try:
            while True:
                yield _sse("system", snapshot())
                await _sleep_or_wake("system", SYSTEM_INTERVAL_S)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("events.system_failed", error=str(exc))
            yield _sse("error", {"detail": "stream ended"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })
