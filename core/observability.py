"""Making this system's failures visible at the moment they happen.

The defining failure mode here is not a crash, it is silence. A week of finding real defects turned up the
same shape every time: a background task that died on a foreign key with nobody to catch it, a relabel job
that ended at frame 13 of 25 and said so only in a log nobody was reading, a `notify()` that swallows every
exception by design, a gold set that shrank from 400 objects to 47 and kept reporting confident numbers, and
a credential that answered 401 to every request while all its unit tests passed. Each one was found by
somebody going looking.

Three pieces, in the order they earn their keep.

**`spawn`** replaces `asyncio.create_task` for the 24 background jobs this API launches. Those exceptions are
not strictly lost: asyncio reports an unretrieved task exception when the task is garbage collected. But it
arrives late, on the asyncio logger rather than the structured one, and carries no run id, no frame id and no
job kind, so what reaches an operator is a traceback with nothing to attach it to. `spawn` attaches a
done-callback that logs with the context the caller already knew and reports to Sentry.

**`capture`** is for the deliberate swallows. Some of them are correct: a failed notification must not roll
back the work it was announcing. What is not correct is that the failure then exists only as a warning line.

**Tracing** covers the long paths (ingest, autolabel, evaluation) where the interesting question is which
stage took the time or which one stopped.

All three are inert without configuration. No DSN means Sentry is never initialised; no OTLP endpoint means
no exporter is installed. The point is that turning them on is a config change, not a code change.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from typing import Any

from core.logging import get_logger

log = get_logger("observability")

_state: dict[str, Any] = {"sentry": False, "tracing": False}


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def init_observability(app: Any | None = None) -> dict:
    """Wire up whatever is configured, and nothing that is not. Safe to call more than once.

    Returns what was actually enabled rather than what was attempted, so a deployment can assert on it
    instead of assuming an env var took effect.
    """
    dsn = _env("SENTRY_DSN")
    if dsn and not _state["sentry"]:
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=dsn,
                environment=_env("LBX_ENV") or "local",
                release=_env("LBX_RELEASE"),
                # Sampled, because a corpus job emits a lot of spans and the useful signal is the shape of a
                # run rather than every one of them.
                traces_sample_rate=float(_env("SENTRY_TRACES_SAMPLE_RATE") or 0.05),
                # This system handles frames of Indian roads under DPDPA. Request bodies and headers can
                # carry object ids, presigned URLs and a media cookie, none of which belong in a third-party
                # error tracker.
                send_default_pii=False,
                max_request_body_size="never",
            )
            _state["sentry"] = True
            log.info("observability.sentry_enabled")
        except Exception as exc:  # noqa: BLE001 - telemetry must never stop the app booting
            log.warning("observability.sentry_failed", error=str(exc))

    endpoint = _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint and not _state["tracing"]:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({
                "service.name": _env("OTEL_SERVICE_NAME") or "labeloxav-api",
                "deployment.environment": _env("LBX_ENV") or "local",
            }))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            if app is not None:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor.instrument_app(app)
            _state["tracing"] = True
            log.info("observability.tracing_enabled", endpoint=endpoint)
        except Exception as exc:  # noqa: BLE001
            log.warning("observability.tracing_failed", error=str(exc))

    return dict(_state)


def capture(exc: BaseException, **context: Any) -> None:
    """Report an exception that the caller is deliberately not raising.

    Some swallows are right: a failed notification must not roll back the work it was announcing. What is
    wrong is that the failure then exists only as a warning line somebody has to go and read.
    """
    log.warning("observability.captured", error=f"{type(exc).__name__}: {exc}", **_flat(context))
    if not _state["sentry"]:
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            for k, v in context.items():
                scope.set_tag(k, str(v)[:200])
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001 - never let the reporter become the failure
        pass


def spawn(coro: Coroutine, *, name: str, **context: Any) -> asyncio.Task:
    """Launch background work whose failure cannot go unattributed.

    `asyncio.create_task` alone loses the association: the exception surfaces when the task is collected, on
    the asyncio logger, without the run id or frame id the caller had in hand at launch. Here the context is
    captured at spawn time and reported with the failure.
    """
    task = asyncio.create_task(coro, name=name)

    def _done(t: asyncio.Task) -> None:
        if t.cancelled():
            log.info("task.cancelled", task=name, **_flat(context))
            return
        exc = t.exception()
        if exc is None:
            log.info("task.finished", task=name, **_flat(context))
            return
        log.error("task.failed", task=name, error=f"{type(exc).__name__}: {exc}", **_flat(context))
        capture(exc, task=name, **context)

    task.add_done_callback(_done)
    _track(task)
    return task


# Strong references to running tasks. asyncio only holds a weak one, so a task nothing else references can be
# garbage collected mid-flight and simply stop, which for a corpus job means silent partial work.
_running: set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> None:
    _running.add(task)
    task.add_done_callback(_running.discard)


def running_tasks() -> list[str]:
    """Names of the background tasks alive in this process, for a health endpoint to report."""
    return sorted(t.get_name() for t in _running if not t.done())


def _flat(context: dict) -> dict:
    """Structured loggers reject nested values; keep every field a short scalar."""
    return {k: (str(v)[:200] if v is not None else None) for k, v in context.items()}


def status() -> dict:
    return {**_state, "background_tasks": running_tasks()}
