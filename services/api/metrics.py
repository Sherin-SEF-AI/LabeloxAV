"""Minimal, dependency-free Prometheus metrics for the API.

Observability was the missing production piece: structured logs told you what happened, but nothing exposed
rates, latencies, or queue depth for a dashboard or an alert. This adds a /metrics endpoint in Prometheus
text-exposition format without pulling in prometheus_client, so it works with the base install and any
Prometheus/Grafana can scrape it.

Two sources of truth:
  - in-process counters/histograms, updated by a middleware on every request (count, latency, in-flight);
  - live gauges sampled at scrape time (review-queue depth, unembedded backlog), so the numbers are current
    rather than a stale snapshot.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

# Latency histogram buckets in seconds (Prometheus convention: cumulative "le" buckets).
_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_lock = Lock()
_req_total: dict[tuple[str, str, int], int] = defaultdict(int)   # (method, route, status) -> count
_req_sum: dict[tuple[str, str], float] = defaultdict(float)       # (method, route) -> total seconds
_req_bucket: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * len(_BUCKETS))
_in_flight = 0


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    with _lock:
        _req_total[(method, route, status)] += 1
        _req_sum[(method, route)] += seconds
        b = _req_bucket[(method, route)]
        for i, edge in enumerate(_BUCKETS):
            if seconds <= edge:
                b[i] += 1


def inc_in_flight(delta: int) -> None:
    global _in_flight
    with _lock:
        _in_flight += delta


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"')


async def render(db_gauges: dict[str, float] | None = None) -> str:
    """Render the current metrics as Prometheus text. db_gauges are sampled live by the caller."""
    lines: list[str] = []

    lines.append("# HELP lbx_http_requests_total Total HTTP requests by method, route, status.")
    lines.append("# TYPE lbx_http_requests_total counter")
    with _lock:
        for (method, route, status), n in sorted(_req_total.items()):
            lines.append(f'lbx_http_requests_total{{method="{_esc(method)}",route="{_esc(route)}",status="{status}"}} {n}')

        lines.append("# HELP lbx_http_request_duration_seconds Request latency.")
        lines.append("# TYPE lbx_http_request_duration_seconds histogram")
        for (method, route), buckets in sorted(_req_bucket.items()):
            cum = 0
            for i, edge in enumerate(_BUCKETS):
                cum += buckets[i]
                lines.append(f'lbx_http_request_duration_seconds_bucket{{method="{_esc(method)}",route="{_esc(route)}",le="{edge}"}} {cum}')
            total = sum(_req_total.get((method, route, s), 0) for s in {k[2] for k in _req_total if k[:2] == (method, route)})
            lines.append(f'lbx_http_request_duration_seconds_bucket{{method="{_esc(method)}",route="{_esc(route)}",le="+Inf"}} {total}')
            lines.append(f'lbx_http_request_duration_seconds_sum{{method="{_esc(method)}",route="{_esc(route)}"}} {_req_sum[(method, route)]:.6f}')
            lines.append(f'lbx_http_request_duration_seconds_count{{method="{_esc(method)}",route="{_esc(route)}"}} {total}')

        lines.append("# HELP lbx_http_requests_in_flight In-flight requests.")
        lines.append("# TYPE lbx_http_requests_in_flight gauge")
        lines.append(f"lbx_http_requests_in_flight {_in_flight}")

    # A TYPE line must carry the bare metric name, not the labels, so emit it once per base name before its
    # series (a labelled TYPE line is rejected by strict scrapers).
    emitted: set[str] = set()
    for name, value in (db_gauges or {}).items():
        base = name.split("{", 1)[0]
        if base not in emitted:
            lines.append(f"# TYPE {base} gauge")
            emitted.add(base)
        lines.append(f"{name} {value}")

    return "\n".join(lines) + "\n"


async def sample_db_gauges() -> dict[str, float]:
    """Live corpus gauges for the scrape: review-queue depth and unembedded backlog. Cheap counts; any error
    yields no gauge rather than failing the scrape."""
    out: dict[str, float] = {}
    try:
        from sqlalchemy import func, select

        from db.models import Object
        from db.session import get_sessionmaker

        async with get_sessionmaker()() as db:
            by_state = dict((await db.execute(
                select(Object.state, func.count()).group_by(Object.state))).all())
        for state, n in by_state.items():
            out[f'lbx_objects_by_state{{state="{_esc(str(state))}"}}'] = float(n)
        out["lbx_review_queue_depth"] = float(by_state.get("review", 0))
    except Exception:  # noqa: BLE001 - a scrape must never 500 the endpoint
        pass
    try:
        from db.session import get_sessionmaker
        from services.intelligence.embed.daemon import pending_counts

        async with get_sessionmaker()() as db:
            p = await pending_counts(db)
        out["lbx_embed_pending_frames"] = float(p["frames"])
        out["lbx_embed_pending_objects"] = float(p["objects"])
    except Exception:  # noqa: BLE001
        pass
    return out


class MetricsMiddleware:
    """ASGI-ish middleware wired via FastAPI's http middleware hook: times each request and records it under
    its route template (not the raw path, so /frame/{id} does not explode the cardinality)."""

    async def __call__(self, request, call_next):
        inc_in_flight(1)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            inc_in_flight(-1)
            route = request.scope.get("route")
            template = getattr(route, "path", None) or request.url.path
            observe_request(request.method, template, status, time.perf_counter() - t0)
