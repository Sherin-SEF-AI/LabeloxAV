"""The provider router: Groq as the fast path, Ollama as the floor, with fallback, a circuit breaker, and
escalate-on-unsure. These are the guards that let the two coexist without the cloud ever being a hard
dependency, so they are worth pinning. Groq is mocked (no network, no key needed)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.paths.path_c_qwen3vl import VlmResult
from services.llm.groq_client import GroqError
from services.llm.router import CircuitBreaker, RoutedVlmClient, make_vlm_client, route_text_json

CROP = np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def vlm_cfg():
    v = get_settings().models.vlm
    g = get_settings().groq
    saved = (v.vision_provider, v.text_provider, v.escalate_provider, v.allow_cloud_media, g.api_key)
    yield v, g
    (v.vision_provider, v.text_provider, v.escalate_provider, v.allow_cloud_media, g.api_key) = saved


def test_breaker_opens_after_threshold_and_reopens_after_cooldown():
    t = {"now": 0.0}
    b = CircuitBreaker(threshold=2, cooldown_s=10, clock=lambda: t["now"])
    assert b.allow()
    b.record_failure(); assert b.allow()      # 1 failure, still closed
    b.record_failure(); assert not b.allow()   # 2 failures, open
    t["now"] = 11                              # past cooldown
    assert b.allow()                           # half-open probe allowed
    b.record_success(); assert b.allow()       # a success closes it


def test_no_cloud_config_returns_plain_ollama(vlm_cfg):
    v, _ = vlm_cfg
    v.vision_provider = "ollama"; v.escalate_provider = None
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient
    assert isinstance(make_vlm_client(), OllamaVlmClient)


def test_groq_success_is_used_and_tagged(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    client = make_vlm_client()
    assert isinstance(client, RoutedVlmClient)

    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: {"class": "rider", "confident": True, "attributes": {}, "caption": "x"})
    monkeypatch.setattr("services.autolabel.paths.path_c_qwen3vl.OllamaVlmClient.verify",
                        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("ollama should not be called")))

    r = client.verify(CROP, ["rider", "pedestrian"], {})
    assert r.class_name == "rider" and r.provider == "groq"


def test_groq_failure_falls_back_to_ollama(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    client = make_vlm_client()

    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: (_ for _ in ()).throw(GroqError("429 rate limit")))
    monkeypatch.setattr("services.autolabel.paths.path_c_qwen3vl.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="pedestrian", confident=True))

    r = client.verify(CROP, ["rider", "pedestrian"], {})
    assert r.class_name == "pedestrian" and r.provider == "ollama"


def test_repeated_groq_failure_opens_circuit_and_stops_calling_it(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    v.__dict__["breaker_threshold"] = 2
    client = make_vlm_client()

    calls = {"groq": 0}
    def boom(self, *a, **k):
        calls["groq"] += 1
        raise GroqError("down")
    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json", boom)
    monkeypatch.setattr("services.autolabel.paths.path_c_qwen3vl.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="bus", confident=True))

    for _ in range(5):
        client.verify(CROP, ["bus"], {})
    # after the breaker opens (2 failures) the cloud is skipped, so groq was called far fewer than 5 times
    assert calls["groq"] <= 3


def test_allow_cloud_media_false_keeps_vision_local(vlm_cfg):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = False; g.api_key = "test-key"
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient
    # media not allowed out -> the router is not even built; vision stays fully local
    assert isinstance(make_vlm_client(), OllamaVlmClient)


def test_escalate_only_fires_on_a_not_confident_verdict(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "ollama"; v.escalate_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    client = make_vlm_client()
    assert isinstance(client, RoutedVlmClient)

    esc = {"n": 0}
    def escalate(self, *a, **k):
        esc["n"] += 1
        return {"class": "autorickshaw", "confident": True, "attributes": {}, "caption": ""}
    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json", escalate)

    # confident primary -> no escalation
    monkeypatch.setattr("services.autolabel.paths.path_c_qwen3vl.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="sedan", confident=True, provider="ollama"))
    client.verify(CROP, ["sedan"], {}); assert esc["n"] == 0

    # not-confident primary -> escalate is consulted and its answer wins
    monkeypatch.setattr("services.autolabel.paths.path_c_qwen3vl.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="sedan", confident=False, provider="ollama"))
    r = client.verify(CROP, ["sedan"], {})
    assert esc["n"] == 1 and r.class_name == "autorickshaw" and r.provider == "groq:escalate"


def test_text_route_uses_groq_then_falls_back_to_ollama(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.text_provider = "groq"; g.api_key = "test-key"

    # groq succeeds -> its answer is used
    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: {"action": "accept", "classes": ["rider"], "conf_min": None})
    assert route_text_json("approve the riders") == {"action": "accept", "classes": ["rider"], "conf_min": None}

    # groq fails AND the local ollama call fails -> None, so nl.py falls back to its rule parser
    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: (_ for _ in ()).throw(GroqError("down")))
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ollama")))
    assert route_text_json("do a thing") is None
