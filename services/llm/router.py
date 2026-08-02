"""Provider routing: cloud providers as the fast or strong path, Ollama as the always-available local floor.

The whole point is that they coexist safely. A router owns the policy so the providers stay dumb and the
call sites (Path C, nl.py) keep calling one interface:

  - availability: no API key, or vision with allow_cloud_media off, means the cloud path is simply not
    taken and ollama serves the call. Local dev works with zero cloud setup.
  - failure fallback: any cloud error (network, 429 rate limit, bad reply) falls through to ollama for that call.
  - circuit breaker: after N consecutive cloud failures the circuit opens and the cloud is skipped for a
    cooldown, so a Groq outage does not add a doomed round-trip to every single call; one half-open probe
    after the cooldown closes it again.
  - escalate: an optional stronger provider is re-asked ONLY when the primary verdict is not confident, so the
    extra cost is spent only on the genuinely hard crops.

Every verdict records which provider produced it (VlmResult.provider), so a gate decision stays traceable and
providers can be compared against each other on the gold set.

Two cloud providers are wired, and they are not interchangeable. Groq is the throughput path for the
autolabel stream. Anthropic is the judge: slower and dearer per crop, used where being right matters more
than being quick, which is the escalate slot and the bulk pre-review of a measurement batch. Adding a third
means writing one client and one line in _CLOUD_VLM, not touching the policy.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from core.config import Settings, get_settings
from core.logging import get_logger
from services.autolabel.paths.path_c_qwen3vl import (
    OllamaVlmClient,
    VlmClient,
    VlmResult,
    _build_prompt,
)
from services.llm.anthropic_client import AnthropicClient, AnthropicError
from services.llm.groq_client import GroqClient, GroqError

log = get_logger("llm.router")

# Any provider error means "fall back", and the router deliberately does not care which cloud raised it.
CloudError = (GroqError, AnthropicError)


class CircuitBreaker:
    """Opens after `threshold` consecutive failures and stays open for `cooldown_s`, then allows one
    half-open probe. Pure and clock-injectable so it is testable without sleeping."""

    def __init__(self, threshold: int, cooldown_s: float, clock=time.monotonic) -> None:
        self.threshold = max(1, threshold)
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._fails = 0
        self._open_until = 0.0

    def allow(self) -> bool:
        if self._fails < self.threshold:
            return True
        if self._clock() >= self._open_until:
            return True  # half-open: let one probe through
        return False

    def record_success(self) -> None:
        self._fails = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._fails += 1
        if self._fails >= self.threshold:
            self._open_until = self._clock() + self.cooldown_s

    @property
    def is_open(self) -> bool:
        return self._fails >= self.threshold and self._clock() < self._open_until


class GroqVlmClient:
    """Vision verifier backed by Groq's multimodal API. Same VlmClient.verify shape as OllamaVlmClient; the
    router decides when to use it. Raises GroqError up to the router (it does not fall back itself)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = GroqClient(self.settings)
        self.model = self.settings.groq.vision_model

    def available(self) -> bool:
        return self.client.available()

    def verify(self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict,
               temperature: float = 0.0) -> VlmResult:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return VlmResult()
        data = self.client.chat_json(_build_prompt(shortlist, attr_schema), model=self.model,
                                     image_jpeg=buf.tobytes(), temperature=temperature)
        return VlmResult(
            class_name=data.get("class"),
            attrs=data.get("attributes", {}) or {},
            caption=data.get("caption", "") or "",
            confident=bool(data.get("confident", False)),
            raw=data, provider="groq",
        )


class AnthropicVlmClient:
    """Vision verifier backed by Anthropic. Same VlmClient.verify shape as the others, so the router treats
    it identically; it raises AnthropicError upward rather than falling back on its own."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AnthropicClient(self.settings)
        self.model = self.settings.anthropic.vision_model

    def available(self) -> bool:
        return self.client.available()

    def verify(self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict,
               temperature: float = 0.0) -> VlmResult:
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            return VlmResult()
        data = self.client.chat_json(_build_prompt(shortlist, attr_schema), model=self.model,
                                     image_jpeg=buf.tobytes(), temperature=temperature)
        return VlmResult(
            class_name=data.get("class"),
            attrs=data.get("attributes", {}) or {},
            caption=data.get("caption", "") or "",
            confident=bool(data.get("confident", False)),
            raw=data, provider="anthropic",
        )


# The cloud vision providers, by the name used in models.vlm.vision_provider / escalate_provider. Ollama is
# absent on purpose: it is the local floor, never a "cloud path" subject to the breaker or to allow_cloud_media.
_CLOUD_VLM = {"groq": GroqVlmClient, "anthropic": AnthropicVlmClient}


def _cloud_vlm(name: str | None, settings: Settings) -> GroqVlmClient | AnthropicVlmClient | None:
    """The configured cloud vision client, or None when it is unnamed, unknown, or has no key.

    Returning None for an unknown name rather than raising is deliberate: a typo in vision_provider degrades
    to the local model instead of taking the pipeline down, and the log line below is how it gets noticed.
    """
    if not name or name == "ollama":
        return None
    cls = _CLOUD_VLM.get(name)
    if cls is None:
        log.warning("router.unknown_vision_provider", provider=name, known=sorted(_CLOUD_VLM))
        return None
    c = cls(settings)
    return c if c.available() else None


class RoutedVlmClient:
    """A VlmClient that tries the configured cloud provider, falls back to the local one, and optionally
    escalates a not-confident verdict to a stronger provider. Implements the same verify() so it drops in
    behind make_vlm_client with no call-site change."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.models.vlm
        self.ollama = OllamaVlmClient(self.settings)
        self._breaker = CircuitBreaker(cfg.breaker_threshold, cfg.breaker_cooldown_s)

        # The cloud primary is used only when configured AND allowed to see media AND a key exists.
        self._primary = _cloud_vlm(cfg.vision_provider, self.settings) if cfg.allow_cloud_media else None

        # The escalate provider is re-asked only on a not-confident verdict, so pointing it at a stronger and
        # dearer model than the primary is the intended configuration rather than a waste.
        self._escalate = _cloud_vlm(cfg.escalate_provider, self.settings) if cfg.allow_cloud_media else None

    def _try_cloud(self, client, *args, **kwargs) -> VlmResult | None:
        """One cloud attempt under the breaker. Returns None on any provider failure, which is the router's
        single signal to use the local floor instead."""
        if not self._breaker.allow():
            return None
        try:
            r = client.verify(*args, **kwargs)
            self._breaker.record_success()
            return r
        except CloudError as exc:
            self._breaker.record_failure()
            log.info("router.cloud_fallback", error=str(exc), open=self._breaker.is_open,
                     provider=type(client).__name__)
            return None

    def verify(self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict,
               temperature: float = 0.0) -> VlmResult:
        res: VlmResult | None = None
        if self._primary is not None:
            res = self._try_cloud(self._primary, crop_bgr, shortlist, attr_schema, temperature)
        if res is None:
            res = self.ollama.verify(crop_bgr, shortlist, attr_schema, temperature)
            if not res.provider:
                res.provider = "ollama"

        # Escalate only a real-but-unsure verdict, so the extra cloud call is spent on the hard crops, not the
        # confident ones or the empty ones.
        if self._escalate is not None and res.class_name and not res.confident:
            esc = self._try_cloud(self._escalate, crop_bgr, shortlist, attr_schema, temperature)
            if esc is not None and esc.class_name:
                esc.provider = f"{esc.provider or 'cloud'}:escalate"
                return esc
        return res


def make_vlm_client(settings: Settings | None = None) -> VlmClient:
    """Return the vision client for the configured providers. When nothing points at the cloud, this is a
    plain OllamaVlmClient (identical to before); otherwise the RoutedVlmClient wraps cloud + local."""
    settings = settings or get_settings()
    cfg = settings.models.vlm
    uses_cloud = cfg.vision_provider in _CLOUD_VLM or cfg.escalate_provider in _CLOUD_VLM
    if uses_cloud and cfg.allow_cloud_media:
        return RoutedVlmClient(settings)
    return OllamaVlmClient(settings)


def route_text_json(prompt: str, *, temperature: float = 0.0, settings: Settings | None = None) -> dict | None:
    """Route a text (no image) JSON chat to Groq when text_provider=groq and a key exists, else Ollama. Returns
    the parsed object, or None on any failure so the caller (nl.py) falls back to its deterministic parser.
    No media ever leaves the box here: this path only sends text."""
    settings = settings or get_settings()
    vcfg = settings.models.vlm

    if vcfg.text_provider == "groq":
        client = GroqClient(settings)
        if client.available():
            try:
                return client.chat_json(prompt, model=settings.groq.text_model, temperature=temperature)
            except GroqError as exc:
                log.info("router.text_cloud_fallback", provider="groq", error=str(exc))
    elif vcfg.text_provider == "anthropic":
        aclient = AnthropicClient(settings)
        if aclient.available():
            try:
                return aclient.chat_json(prompt, model=settings.anthropic.text_model, temperature=temperature)
            except AnthropicError as exc:
                log.info("router.text_cloud_fallback", provider="anthropic", error=str(exc))

    # Local fallback: the same Ollama text call nl.py used before (text-only, no images).
    import json

    import httpx

    try:
        payload = {"model": vcfg.ollama_tag, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": temperature}}
        resp = httpx.post(f"{vcfg.ollama_url}/api/chat", json=payload,
                          timeout=min(getattr(vcfg, "timeout_s", 20), 20))
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])
    except Exception:  # noqa: BLE001
        return None
