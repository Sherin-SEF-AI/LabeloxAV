"""Every model call is counted, costed where a price is known, and capped where a cap is set.

Two cloud providers were wired in and nothing counted a token. These pin the properties that make the
ledger worth trusting: a call is recorded whether it succeeds or fails; an unpriced cloud model reads as
unknown rather than free; counters survive a database outage; and the cap really does send work to the
local model.
"""

from __future__ import annotations

import httpx
import pytest

from core.config import LlmPrice, get_settings
from services.llm import ledger

pytestmark = pytest.mark.db


class _Reply:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    import psycopg

    ledger.reset_for_tests()
    pg = get_settings().postgres
    with psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password,
                         dbname=pg.db) as conn:
        conn.execute("DELETE FROM llm_usage_daily")
        conn.commit()
    monkeypatch.setattr(get_settings().llm, "flush_interval_s", 3600.0)   # flush only when a test says so
    monkeypatch.setattr(get_settings().llm, "prices", {})
    monkeypatch.setattr(get_settings().llm, "daily_budget_usd", 0.0)
    yield
    ledger.reset_for_tests()


def _rows() -> dict:
    import psycopg

    pg = get_settings().postgres
    with psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password,
                         dbname=pg.db) as conn:
        cur = conn.execute("SELECT provider, model, purpose, calls, failures, input_tokens, output_tokens, "
                           "images, unmetered_calls, cost_usd FROM llm_usage_daily")
        return {(r[0], r[1], r[2]): dict(zip(("calls", "failures", "in", "out", "images", "unmetered", "cost"),
                                             r[3:], strict=True)) for r in cur.fetchall()}


@pytest.mark.parametrize("provider,body,expected", [
    ("groq", {"usage": {"prompt_tokens": 120, "completion_tokens": 30}}, (120, 30)),
    ("anthropic", {"usage": {"input_tokens": 400, "output_tokens": 55}}, (400, 55)),
    ("ollama", {"prompt_eval_count": 900, "eval_count": 12, "message": {}}, (900, 12)),
    ("ollama", {"message": {"content": "{}"}}, None),
    ("groq", "not a dict", None),
])
def test_token_counts_are_read_from_each_providers_reply_shape(provider, body, expected):
    assert ledger.usage_from_reply(provider, body) == expected


def test_a_successful_call_is_recorded_with_its_tokens_and_images(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Reply({"prompt_eval_count": 700, "eval_count": 20}))
    payload = {"messages": [{"role": "user", "content": "x", "images": ["b64", "b64"]}]}
    ledger.post("http://ollama/api/chat", provider="ollama", model="qwen2.5vl:7b", purpose="vlm_verify",
                json=payload, timeout=5)
    assert ledger.flush() == 1
    row = _rows()[("ollama", "qwen2.5vl:7b", "vlm_verify")]
    assert (row["calls"], row["failures"], row["in"], row["out"], row["images"]) == (1, 0, 700, 20, 2)
    assert row["cost"] == 0.0, "a local model costs nothing, and that is a known zero"


def test_a_transport_failure_is_counted_and_still_raised(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(httpx.ConnectError):
        ledger.post("http://x", provider="groq", model="m", purpose="judge", json={}, timeout=1)
    ledger.flush()
    row = _rows()[("groq", "m", "judge")]
    assert (row["calls"], row["failures"]) == (1, 1)


def test_an_error_status_is_a_failure_and_the_response_is_returned_untouched(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Reply({"error": "rate"}, status=429))
    resp = ledger.post("http://x", provider="groq", model="m", purpose="judge", json={}, timeout=1)
    assert resp.status_code == 429, "the caller's own retry logic must still see the response"
    ledger.flush()
    assert _rows()[("groq", "m", "judge")]["failures"] == 1


def test_an_unpriced_cloud_model_reads_as_unknown_cost_not_free(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _Reply({"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}))
    ledger.post("http://x", provider="groq", model="unpriced", purpose="judge", json={}, timeout=1)
    ledger.flush()
    row = _rows()[("groq", "unpriced", "judge")]
    assert row["in"] == 1000 and row["cost"] is None


def test_a_priced_model_is_costed_from_its_tokens(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "prices",
                        {"anthropic:judge-model": LlmPrice(input_per_mtok=3.0, output_per_mtok=15.0)})
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _Reply({"usage": {"input_tokens": 2_000_000, "output_tokens": 100_000}}))
    ledger.post("http://x", provider="anthropic", model="judge-model", purpose="judge", json={}, timeout=1)
    ledger.post("http://x", provider="anthropic", model="judge-model", purpose="judge", json={}, timeout=1)
    ledger.flush()
    row = _rows()[("anthropic", "judge-model", "judge")]
    assert row["calls"] == 2
    assert row["cost"] == pytest.approx(2 * (2.0 * 3.0 + 0.1 * 15.0))


def test_a_reply_with_no_token_counts_is_marked_unmetered(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Reply({"message": {"content": "{}"}}))
    ledger.post("http://x", provider="ollama", model="m", purpose="sign_ocr", json={}, timeout=1)
    ledger.flush()
    assert _rows()[("ollama", "m", "sign_ocr")]["unmetered"] == 1


def test_two_flushes_add_rather_than_overwrite(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Reply({"prompt_eval_count": 10, "eval_count": 1}))
    for _ in range(2):
        ledger.post("http://x", provider="ollama", model="m", purpose="nl", json={}, timeout=1)
        ledger.flush()
    row = _rows()[("ollama", "m", "nl")]
    assert (row["calls"], row["in"]) == (2, 20)


def test_counters_survive_a_database_that_is_down(monkeypatch):
    ledger.record("ollama", "m", "nl", input_tokens=5, output_tokens=1)

    def down():
        raise OSError("connection refused")

    real = ledger._connect
    monkeypatch.setattr(ledger, "_connect", down)
    assert ledger.flush() == 0, "a failed write must not claim to have written"
    monkeypatch.setattr(ledger, "_connect", real)
    assert ledger.flush() == 1
    assert _rows()[("ollama", "m", "nl")]["in"] == 5, "the call recorded during the outage was lost"


def test_recording_never_raises_into_the_model_call(monkeypatch):
    monkeypatch.setattr(ledger, "cost_of", lambda *a, **k: 1 / 0)
    ledger.record("groq", "m", "judge", input_tokens=1, output_tokens=1)   # must not raise


def test_the_cap_counts_spend_not_yet_written_and_spend_already_written(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "prices", {"groq:m": LlmPrice(input_per_mtok=1.0, output_per_mtok=1.0)})
    monkeypatch.setattr(get_settings().llm, "daily_budget_usd", 3.0)
    assert not ledger.cloud_budget_exhausted()
    ledger.record("groq", "m", "judge", input_tokens=2_000_000, output_tokens=0)     # $2, in memory
    assert not ledger.cloud_budget_exhausted()
    ledger.flush()                                                                   # $2, in the database
    ledger.record("groq", "m", "judge", input_tokens=1_500_000, output_tokens=0)     # +$1.50, in memory
    ledger._flushed_cloud_today = None                                               # force a fresh read
    assert ledger.cloud_spend_today() == pytest.approx(3.5)
    assert ledger.cloud_budget_exhausted()


def test_local_calls_never_count_toward_the_cloud_cap(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "daily_budget_usd", 0.01)
    ledger.record("ollama", "m", "vlm_verify", input_tokens=10**9, output_tokens=10**9)
    assert not ledger.cloud_budget_exhausted()


def test_no_cap_means_never_exhausted(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "prices", {"groq:m": LlmPrice(input_per_mtok=1e6)})
    ledger.record("groq", "m", "judge", input_tokens=10**9)
    assert not ledger.cloud_budget_exhausted()


def test_over_the_cap_the_router_answers_from_the_local_model(monkeypatch):
    import numpy as np

    from services.autolabel.paths.path_c_vlm import VlmResult
    from services.llm import router

    monkeypatch.setattr(ledger, "cloud_budget_exhausted", lambda: True)
    monkeypatch.setattr(ledger, "cloud_spend_today", lambda **k: 9.0)
    asked = {"cloud": 0, "local": 0}

    class _Cloud:
        def verify(self, *a, **k):
            asked["cloud"] += 1
            return VlmResult(class_name="car", confident=True, provider="groq")

    class _Local:
        def verify(self, *a, **k):
            asked["local"] += 1
            return VlmResult(class_name="car", confident=True)

    r = router.RoutedVlmClient.__new__(router.RoutedVlmClient)
    r.settings = get_settings()
    r._breaker = router.CircuitBreaker(3, 30.0)
    r._budget_logged = False
    r._primary, r._escalate, r.ollama = _Cloud(), None, _Local()
    res = r.verify(np.zeros((8, 8, 3), np.uint8), ["car"], {})
    assert asked == {"cloud": 0, "local": 1} and res.provider == "ollama"


def test_no_model_call_site_posts_without_the_ledger():
    """A call site that posts to a model on its own is a call nobody can see."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "services"
    model_url = re.compile(r"/api/chat|/api/generate|/chat/completions|base_url[^\n]*/messages")
    offenders = []
    for f in root.rglob("*.py"):
        if f.name == "ledger.py":
            continue
        src = f.read_text()
        for m in re.finditer(r"httpx\.post\(", src):
            window = src[m.start():m.start() + 260]
            if model_url.search(window):
                offenders.append(f"{f.relative_to(root.parent)}:{src[:m.start()].count(chr(10)) + 1}")
    assert offenders == [], f"model calls that bypass services/llm/ledger.post: {offenders}"
