"""The ledger of model calls: who asked which model for what, how many tokens, how long, what it cost.

Groq and Anthropic are both wired in, and until this existed nothing counted a token. The only budget was
a call counter two agents kept in memory, and several call sites posted straight to a model with no record
that they had. A provider invoice was the first place anyone could learn what a sweep had cost.

Every call goes through `post` (or reports itself with `record`) and lands here as an increment to one
daily row per provider, model and purpose. The call sites are synchronous, run in worker threads, and live
in more than one process (the API, the GPU worker, scripts), so the counters are kept in memory under a
lock and written with a plain synchronous connection every few seconds, on demand, and at exit. A database
that is down never fails a model call: the counters are simply kept until the next write succeeds.

Cost is tokens times the configured price (`llm.prices`). A cloud model with no price is recorded with a
NULL cost rather than zero, because an unpriced model must not read as a free one. A local model is zero.

`cloud_budget_exhausted()` is what lets the router stop sending work to the cloud once today's spend
reaches `llm.daily_budget_usd`; the call then falls back to the local model, the same path it takes when a
provider is down.
"""

from __future__ import annotations

import atexit
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import httpx

from core.config import get_settings
from core.logging import get_logger

log = get_logger("llm.ledger")

LOCAL_PROVIDERS = frozenset({"ollama", "llamacpp"})
Key = tuple[date, str, str, str]


@dataclass
class _Counts:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    latency_ms: int = 0
    unmetered_calls: int = 0
    cost_usd: float | None = None

    def add(self, other: _Counts) -> None:
        self.calls += other.calls
        self.failures += other.failures
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.images += other.images
        self.latency_ms += other.latency_ms
        self.unmetered_calls += other.unmetered_calls
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd


_lock = threading.Lock()
_pending: dict[Key, _Counts] = {}
_last_flush = 0.0
_flushed_cloud_today: tuple[date, float, float] | None = None   # (day, dollars, read at)
_atexit_registered = False


def today() -> date:
    return datetime.now(UTC).date()


def usage_from_reply(provider: str, body: Any) -> tuple[int, int] | None:
    """(input tokens, output tokens) as the provider reported them, or None when it reported none.

    Three shapes: OpenAI-compatible (`usage.prompt_tokens`), Anthropic (`usage.input_tokens`) and Ollama
    (`prompt_eval_count` at the top level).
    """
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if isinstance(usage, dict):
        i = usage.get("prompt_tokens", usage.get("input_tokens"))
        o = usage.get("completion_tokens", usage.get("output_tokens"))
        if i is not None or o is not None:
            return int(i or 0), int(o or 0)
    if "prompt_eval_count" in body or "eval_count" in body:
        return int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)
    return None


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    """Dollars per million (input, output) tokens. None when the model is a cloud model with no price."""
    if provider in LOCAL_PROVIDERS:
        return 0.0, 0.0
    prices = get_settings().llm.prices
    p = prices.get(f"{provider}:{model}") or prices.get(provider)
    if p is None:
        return None
    return float(p.input_per_mtok), float(p.output_per_mtok)


def cost_of(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(provider, model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000.0


def record(provider: str, model: str, purpose: str, *, input_tokens: int | None = None,
           output_tokens: int | None = None, images: int = 0, latency_ms: float = 0.0,
           ok: bool = True) -> None:
    """Count one call. Never raises: accounting must not be the reason a model call fails."""
    try:
        if not get_settings().llm.enabled:
            return
        metered = input_tokens is not None or output_tokens is not None
        i, o = int(input_tokens or 0), int(output_tokens or 0)
        c = _Counts(calls=1, failures=0 if ok else 1, input_tokens=i, output_tokens=o, images=int(images),
                    latency_ms=int(latency_ms), unmetered_calls=0 if (metered or not ok) else 1,
                    cost_usd=cost_of(provider, model, i, o) if metered else
                    (0.0 if provider in LOCAL_PROVIDERS else None))
        key: Key = (today(), provider[:32], (model or "unknown")[:128], (purpose or "unspecified")[:48])
        global _atexit_registered
        with _lock:
            _pending.setdefault(key, _Counts()).add(c)
            due = time.monotonic() - _last_flush >= float(get_settings().llm.flush_interval_s)
            if not _atexit_registered:
                atexit.register(flush)
                _atexit_registered = True
        if due:
            flush()
    except Exception as exc:  # noqa: BLE001
        log.warning("ledger.record_failed", error=str(exc)[:160])


def _count_images(payload: Any) -> int:
    """Images in a chat payload, across the three providers' message shapes."""
    n = 0
    try:
        for m in (payload or {}).get("messages", []):
            n += len(m.get("images") or [])
            content = m.get("content")
            if isinstance(content, list):
                n += sum(1 for part in content if isinstance(part, dict)
                         and part.get("type") in ("image", "image_url"))
    except Exception:  # noqa: BLE001
        return n
    return n


def post(url: str, *, provider: str, model: str, purpose: str, json: Any = None,
         headers: dict[str, str] | None = None, timeout: float | None = None) -> httpx.Response:
    """`httpx.post` for a model call, recorded. Returns the response exactly as httpx would, and raises
    exactly what httpx would; a transport failure and an HTTP error status are both counted as failures.

    Every direct model call in the codebase goes through this, which is what makes the ledger complete:
    a call site that posted on its own was a call nobody could see.
    """
    t0 = time.perf_counter()
    images = _count_images(json)
    try:
        kwargs: dict[str, Any] = {"json": json, "timeout": timeout}
        if headers is not None:
            kwargs["headers"] = headers
        resp = httpx.post(url, **kwargs)
    except Exception:
        record(provider, model, purpose, images=images, latency_ms=(time.perf_counter() - t0) * 1000, ok=False)
        raise
    ms = (time.perf_counter() - t0) * 1000
    status = getattr(resp, "status_code", 200)
    ok = not (isinstance(status, int) and status >= 400)
    usage = None
    if ok:
        try:
            usage = usage_from_reply(provider, resp.json())
        except Exception:  # noqa: BLE001 - a reply that is not JSON is the caller's problem, not the ledger's
            usage = None
    record(provider, model, purpose, input_tokens=usage[0] if usage else None,
           output_tokens=usage[1] if usage else None, images=images, latency_ms=ms, ok=ok)
    return resp


_UPSERT = """
INSERT INTO llm_usage_daily (day, provider, model, purpose, calls, failures, input_tokens, output_tokens,
                             images, latency_ms, unmetered_calls, cost_usd, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (day, provider, model, purpose) DO UPDATE SET
    calls = llm_usage_daily.calls + EXCLUDED.calls,
    failures = llm_usage_daily.failures + EXCLUDED.failures,
    input_tokens = llm_usage_daily.input_tokens + EXCLUDED.input_tokens,
    output_tokens = llm_usage_daily.output_tokens + EXCLUDED.output_tokens,
    images = llm_usage_daily.images + EXCLUDED.images,
    latency_ms = llm_usage_daily.latency_ms + EXCLUDED.latency_ms,
    unmetered_calls = llm_usage_daily.unmetered_calls + EXCLUDED.unmetered_calls,
    cost_usd = CASE WHEN EXCLUDED.cost_usd IS NULL THEN llm_usage_daily.cost_usd
                    ELSE COALESCE(llm_usage_daily.cost_usd, 0) + EXCLUDED.cost_usd END,
    updated_at = now()
"""


def _connect() -> Any:
    import psycopg

    pg = get_settings().postgres
    return psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password, dbname=pg.db,
                           connect_timeout=3)


def flush() -> int:
    """Write the pending counters. Returns how many rows were written. Never raises.

    The counters are taken out under the lock and put back if the write fails, so a call recorded while
    the database was down is written once it is back, and one recorded during the write is not lost.
    """
    global _last_flush
    with _lock:
        if not _pending:
            _last_flush = time.monotonic()
            return 0
        batch = dict(_pending)
        _pending.clear()
        _last_flush = time.monotonic()
    try:
        with _connect() as conn, conn.cursor() as cur:
            for (day, provider, model, purpose), c in batch.items():
                cur.execute(_UPSERT, (day, provider, model, purpose, c.calls, c.failures, c.input_tokens,
                                      c.output_tokens, c.images, c.latency_ms, c.unmetered_calls, c.cost_usd))
            conn.commit()
        return len(batch)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            for key, c in batch.items():
                _pending.setdefault(key, _Counts()).add(c)
        log.warning("ledger.flush_failed", error=str(exc)[:160], kept=len(batch))
        return 0


def pending_cloud_spend(day: date | None = None) -> float:
    day = day or today()
    with _lock:
        return sum(c.cost_usd or 0.0 for (d, provider, _m, _p), c in _pending.items()
                   if d == day and provider not in LOCAL_PROVIDERS)


def cloud_spend_today(*, max_age_s: float = 30.0) -> float:
    """Today's recorded cloud spend in dollars: what is in the database plus what has not been written yet.

    The database half is re-read at most every `max_age_s`; this is asked before every routed call, and a
    query per crop would cost more attention than the budget is worth.
    """
    global _flushed_cloud_today
    day = today()
    cached = _flushed_cloud_today
    if cached is None or cached[0] != day or time.monotonic() - cached[2] > max_age_s:
        stored = cached[1] if cached and cached[0] == day else 0.0
        try:
            with _connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage_daily "
                            "WHERE day = %s AND provider <> ALL(%s)", (day, list(LOCAL_PROVIDERS)))
                stored = float(cur.fetchone()[0] or 0.0)
        except Exception as exc:  # noqa: BLE001 - keep the last known figure rather than read as zero
            log.warning("ledger.spend_read_failed", error=str(exc)[:160])
        _flushed_cloud_today = (day, stored, time.monotonic())
        cached = _flushed_cloud_today
    return cached[1] + pending_cloud_spend(day)


def cloud_budget_exhausted() -> bool:
    """Whether today's cloud spend has reached the configured cap. False when there is no cap."""
    cap = float(get_settings().llm.daily_budget_usd or 0.0)
    if cap <= 0:
        return False
    return cloud_spend_today() >= cap


def reset_for_tests() -> None:
    global _last_flush, _flushed_cloud_today
    with _lock:
        _pending.clear()
        # As if a write had just happened, so the next record waits out the interval instead of writing
        # at once and a test decides for itself when the counters reach the database.
        _last_flush = time.monotonic()
        _flushed_cloud_today = None
