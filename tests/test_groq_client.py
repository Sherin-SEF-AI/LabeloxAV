"""The Groq client's two defences against falling back to the local model when it need not.

Both come from a calibration run against Groq: 21 of 260 crops were refused for being under 32 pixels on a
side, and a burst hit the tokens-per-minute limit with Groq asking for a wait of a few hundred milliseconds.
Each refusal would otherwise have handed the call to the local vision model. No network: httpx is stubbed.
"""

from __future__ import annotations

import cv2
import httpx
import numpy as np
import pytest

from services.llm import groq_client as G


def _jpeg(w: int, h: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def _size(jpeg: bytes) -> tuple[int, int]:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


def test_tiny_crop_is_enlarged_keeping_its_shape():
    w, h = _size(G.ensure_min_side(_jpeg(10, 20)))
    assert min(w, h) >= G.MIN_SIDE_PX
    assert abs(h / w - 2.0) < 0.1


def test_large_crop_passes_through_unchanged():
    big = _jpeg(200, 120)
    assert G.ensure_min_side(big) == big


class _Resp:
    def __init__(self, status: int, body: dict | str, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else __import__("json").dumps(body)

    def json(self):
        return self._body


def _client(monkeypatch, responses: list[_Resp], sent: list | None = None) -> G.GroqClient:
    c = G.GroqClient()
    c.cfg = c.cfg.model_copy(update={"api_key": "test-key"})

    def post(url, json, headers, timeout):
        if sent is not None:
            sent.append(json)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    return c


OK = _Resp(200, {"choices": [{"message": {"content": '{"verdict": "correct"}'}}]})


def test_short_rate_limit_is_waited_out_not_failed(monkeypatch):
    limited = _Resp(400, {"error": {"code": "rate_limit_exceeded",
                                    "message": "Rate limit reached ... Please try again in 435.12ms."}})
    c = _client(monkeypatch, [limited, OK])
    assert c.chat_json("q", model="m") == {"verdict": "correct"}


def test_429_with_retry_after_header_is_retried(monkeypatch):
    c = _client(monkeypatch, [_Resp(429, "slow down", {"retry-after": "0.2"}), OK])
    assert c.chat_json("q", model="m") == {"verdict": "correct"}


def test_long_rate_limit_raises_so_the_router_can_fall_back(monkeypatch):
    c = _client(monkeypatch, [_Resp(429, "slow down", {"retry-after": "60"})])
    with pytest.raises(G.GroqError):
        c.chat_json("q", model="m")


def test_other_errors_raise_with_groqs_message(monkeypatch):
    bad = _Resp(400, {"error": {"message": "Image must have at least 32 pixels in each dimension"}})
    c = _client(monkeypatch, [bad])
    with pytest.raises(G.GroqError, match="32 pixels"):
        c.chat_json("q", model="m")


def test_the_image_sent_is_the_enlarged_one(monkeypatch):
    sent: list = []
    c = _client(monkeypatch, [OK], sent)
    c.chat_json("q", model="m", image_jpeg=_jpeg(8, 8))
    url = sent[0]["messages"][0]["content"][1]["image_url"]["url"]
    import base64
    assert min(_size(base64.b64decode(url.split(",", 1)[1]))) >= G.MIN_SIDE_PX
