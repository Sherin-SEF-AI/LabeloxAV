"""The provider router: Groq as the fast path, Ollama as the floor, with fallback, a circuit breaker, and
escalate-on-unsure. These are the guards that let the two coexist without the cloud ever being a hard
dependency, so they are worth pinning. Groq is mocked (no network, no key needed)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.paths.path_c_vlm import VlmResult
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
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient
    assert isinstance(make_vlm_client(), OllamaVlmClient)


def test_groq_success_is_used_and_tagged(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    client = make_vlm_client()
    assert isinstance(client, RoutedVlmClient)

    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: {"class": "rider", "confident": True, "attributes": {}, "caption": "x"})
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
                        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("ollama should not be called")))

    r = client.verify(CROP, ["rider", "pedestrian"], {})
    assert r.class_name == "rider" and r.provider == "groq"


def test_groq_failure_falls_back_to_ollama(vlm_cfg, monkeypatch):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = True; g.api_key = "test-key"
    client = make_vlm_client()

    monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                        lambda self, *a, **k: (_ for _ in ()).throw(GroqError("429 rate limit")))
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
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
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="bus", confident=True))

    for _ in range(5):
        client.verify(CROP, ["bus"], {})
    # after the breaker opens (2 failures) the cloud is skipped, so groq was called far fewer than 5 times
    assert calls["groq"] <= 3


def test_allow_cloud_media_false_keeps_vision_local(vlm_cfg):
    v, g = vlm_cfg
    v.vision_provider = "groq"; v.allow_cloud_media = False; g.api_key = "test-key"
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient
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
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
                        lambda self, *a, **k: VlmResult(class_name="sedan", confident=True, provider="ollama"))
    client.verify(CROP, ["sedan"], {}); assert esc["n"] == 0

    # not-confident primary -> escalate is consulted and its answer wins
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
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


# --- the frontier judge provider -------------------------------------------------------------
#
# Groq is the throughput path; Anthropic is wired for the job Groq is the wrong shape for, deciding whether
# an existing label is right on a small crop. The router must treat it exactly as it treats Groq, or the
# cloud stops being optional and a key outage becomes an outage.


@pytest.fixture
def anthropic_cfg():
    v = get_settings().models.vlm
    a = get_settings().anthropic
    saved = (v.vision_provider, v.text_provider, v.escalate_provider, v.allow_cloud_media, a.api_key)
    yield v, a
    (v.vision_provider, v.text_provider, v.escalate_provider, v.allow_cloud_media, a.api_key) = saved


def test_anthropic_success_is_used_and_tagged(anthropic_cfg, monkeypatch):
    v, a = anthropic_cfg
    v.vision_provider = "anthropic"; v.allow_cloud_media = True; a.api_key = "test-key"
    client = make_vlm_client()
    assert isinstance(client, RoutedVlmClient)

    monkeypatch.setattr("services.llm.anthropic_client.AnthropicClient.chat_json",
                        lambda self, *a_, **k: {"class": "autorickshaw", "confident": True,
                                                "attributes": {}, "caption": "three-wheeler"})
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
                        lambda self, *a_, **k: (_ for _ in ()).throw(AssertionError("ollama should not be called")))

    r = client.verify(CROP, ["autorickshaw", "e_auto"], {})
    assert r.class_name == "autorickshaw" and r.provider == "anthropic"


def test_anthropic_failure_falls_back_to_ollama(anthropic_cfg, monkeypatch):
    """The property that keeps the cloud optional. Without it, adding a provider adds an outage mode."""
    from services.llm.anthropic_client import AnthropicError

    v, a = anthropic_cfg
    v.vision_provider = "anthropic"; v.allow_cloud_media = True; a.api_key = "test-key"
    client = make_vlm_client()

    monkeypatch.setattr("services.llm.anthropic_client.AnthropicClient.chat_json",
                        lambda self, *a_, **k: (_ for _ in ()).throw(AnthropicError("anthropic http 429")))
    monkeypatch.setattr("services.autolabel.paths.path_c_vlm.OllamaVlmClient.verify",
                        lambda self, *a_, **k: VlmResult(class_name="rider", confident=True, provider="ollama"))

    r = client.verify(CROP, ["rider"], {})
    assert r.class_name == "rider" and r.provider == "ollama"


def test_no_key_means_the_provider_is_simply_not_configured(anthropic_cfg):
    """Naming a provider without a key must not select it, or local dev breaks on someone else's config."""
    v, a = anthropic_cfg
    v.vision_provider = "anthropic"; v.escalate_provider = None; v.allow_cloud_media = True; a.api_key = ""
    client = make_vlm_client()
    # RoutedVlmClient is still returned (the provider is named), but it holds no cloud primary and so
    # behaves as the local floor.
    assert isinstance(client, RoutedVlmClient) and client._primary is None


def test_an_unknown_provider_degrades_to_local_instead_of_crashing(anthropic_cfg):
    """A typo in vision_provider should cost quality, not availability.

    The previous router compared against the literal "groq", so an unrecognised name silently meant "no
    cloud". Now that the name is a dictionary lookup, an unknown key has to be handled explicitly or it
    raises inside the constructor and takes down every Path C call.
    """
    v, a = anthropic_cfg
    v.vision_provider = "anthropik"; v.escalate_provider = None; v.allow_cloud_media = True; a.api_key = "k"
    client = RoutedVlmClient()
    assert client._primary is None


def test_data_residency_still_wins_over_a_configured_frontier_provider(anthropic_cfg):
    """allow_cloud_media=False is a DPDPA posture, not a preference: no crop leaves the box whatever the
    provider says."""
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient

    v, a = anthropic_cfg
    v.vision_provider = "anthropic"; v.escalate_provider = None; v.allow_cloud_media = False; a.api_key = "k"
    assert isinstance(make_vlm_client(), OllamaVlmClient)


def test_escalate_can_be_a_different_provider_than_the_primary(anthropic_cfg, monkeypatch):
    """The configuration this generalisation exists for: fast provider in front, strong one on the hard crop."""
    v, a = anthropic_cfg
    g = get_settings().groq
    saved_g = g.api_key
    try:
        v.vision_provider = "groq"; v.escalate_provider = "anthropic"
        v.allow_cloud_media = True; a.api_key = "ak"; g.api_key = "gk"
        client = RoutedVlmClient()

        monkeypatch.setattr("services.llm.groq_client.GroqClient.chat_json",
                            lambda self, *a_, **k: {"class": "e_auto", "confident": False, "attributes": {}})
        monkeypatch.setattr("services.llm.anthropic_client.AnthropicClient.chat_json",
                            lambda self, *a_, **k: {"class": "autorickshaw", "confident": True, "attributes": {}})

        r = client.verify(CROP, ["autorickshaw", "e_auto"], {})
        # the unsure groq verdict is escalated to anthropic, and the provider tag records both facts
        assert r.class_name == "autorickshaw" and r.provider == "anthropic:escalate"
    finally:
        g.api_key = saved_g
