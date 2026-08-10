"""Client-level behaviour: the Messages API shape differs from OpenAI's, and getting it wrong is silent."""
from __future__ import annotations

import pytest

from core.config import get_settings
from services.llm.anthropic_client import AnthropicClient, AnthropicError, _unwrap_json


def test_no_key_means_unavailable_rather_than_a_failed_call():
    s = get_settings(); saved = s.anthropic.api_key
    try:
        s.anthropic.api_key = ""
        c = AnthropicClient(s)
        assert not c.available()
        with pytest.raises(AnthropicError):
            c.chat_json("hi", model="m")
    finally:
        s.anthropic.api_key = saved


def test_a_fenced_reply_is_still_parsed():
    """Claude answers a 'JSON only' instruction with bare JSON, but fences it under temperature often enough
    that losing a judged crop to two backticks would be a silly failure."""
    assert _unwrap_json('```json\n{"class": "cattle"}\n```') == {"class": "cattle"}
    assert _unwrap_json('{"class": "cattle"}') == {"class": "cattle"}


def test_the_image_goes_in_a_source_block_not_a_data_url(monkeypatch):
    """The OpenAI shape (image_url with a data: URL) is silently ignored by the Messages API, so the call
    would succeed and the model would answer without ever seeing the crop."""
    captured = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"content": [{"type": "text", "text": '{"class":"rider"}'}]}

    def _post(url, json=None, headers=None, timeout=None):
        captured["url"], captured["json"], captured["headers"] = url, json, headers
        return _Resp()

    monkeypatch.setattr("services.llm.anthropic_client.httpx.post", _post)
    s = get_settings(); saved = s.anthropic.api_key
    try:
        s.anthropic.api_key = "k"
        out = AnthropicClient(s).chat_json("prompt", model="m", image_jpeg=b"\xff\xd8jpeg")
    finally:
        s.anthropic.api_key = saved

    assert out == {"class": "rider"}
    assert captured["url"].endswith("/messages")
    assert captured["headers"]["x-api-key"] == "k" and "anthropic-version" in captured["headers"]
    parts = captured["json"]["messages"][0]["content"]
    img = next(p for p in parts if p["type"] == "image")
    assert img["source"]["type"] == "base64" and img["source"]["media_type"] == "image/jpeg"
    assert parts.index(img) < next(i for i, p in enumerate(parts) if p["type"] == "text")


def test_an_http_error_names_the_status_so_a_bad_key_is_distinguishable_from_a_rate_limit(monkeypatch):
    import httpx

    class _Resp:
        status_code = 429
        def raise_for_status(self): raise httpx.HTTPStatusError("x", request=None, response=self)

    monkeypatch.setattr("services.llm.anthropic_client.httpx.post", lambda *a, **k: _Resp())
    s = get_settings(); saved = s.anthropic.api_key
    try:
        s.anthropic.api_key = "k"
        with pytest.raises(AnthropicError, match="429"):
            AnthropicClient(s).chat_json("p", model="m")
    finally:
        s.anthropic.api_key = saved


def test_an_empty_reply_is_an_error_not_an_empty_verdict(monkeypatch):
    """Returning {} would look like 'the judge said nothing is wrong', which is the dangerous reading."""
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"content": []}

    monkeypatch.setattr("services.llm.anthropic_client.httpx.post", lambda *a, **k: _Resp())
    s = get_settings(); saved = s.anthropic.api_key
    try:
        s.anthropic.api_key = "k"
        with pytest.raises(AnthropicError):
            AnthropicClient(s).chat_json("p", model="m")
    finally:
        s.anthropic.api_key = saved
