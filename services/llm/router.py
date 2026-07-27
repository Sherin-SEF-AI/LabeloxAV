"""Provider routing: Groq as the fast cloud path, Ollama as the always-available local floor.

The whole point is that the two coexist safely. A router owns the policy so the providers stay dumb and the
call sites (Path C, nl.py) keep calling one interface:

  - availability: no GROQ_API_KEY, or vision with allow_cloud_media off, means the cloud path is simply not
    taken and ollama serves the call. Local dev works with zero Groq setup.
  - failure fallback: any GroqError (network, 429 rate limit, bad reply) falls through to ollama for that call.
  - circuit breaker: after N consecutive cloud failures the circuit opens and the cloud is skipped for a
    cooldown, so a Groq outage does not add a doomed round-trip to every single call; one half-open probe
    after the cooldown closes it again.
  - escalate: an optional stronger provider is re-asked ONLY when the primary verdict is not confident, so the
    extra cost is spent only on the genuinely hard crops.

Every verdict records which provider produced it (VlmResult.provider), so a gate decision stays traceable and
Groq-vs-Ollama quality can be compared on the gold set.
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
from services.llm.groq_client import GroqClient, GroqError

log = get_logger("llm.router")


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
        self._groq: GroqVlmClient | None = None
        if cfg.vision_provider == "groq" and cfg.allow_cloud_media:
            g = GroqVlmClient(self.settings)
            self._groq = g if g.available() else None

        # The escalate provider is currently groq (a stronger vision model can be pointed at groq.vision_model
        # per deployment). Only wired when distinct from the primary and media egress is allowed.
        self._escalate: GroqVlmClient | None = None
        if cfg.escalate_provider == "groq" and cfg.allow_cloud_media:
            e = GroqVlmClient(self.settings)
            self._escalate = e if e.available() else None

    def _try_groq(self, client: GroqVlmClient, *args, **kwargs) -> VlmResult | None:
        if not self._breaker.allow():
            return None
        try:
            r = client.verify(*args, **kwargs)
            self._breaker.record_success()
            return r
        except GroqError as exc:
            self._breaker.record_failure()
            log.info("router.groq_fallback", error=str(exc), open=self._breaker.is_open)
            return None

    def verify(self, crop_bgr: np.ndarray, shortlist: list[str], attr_schema: dict,
               temperature: float = 0.0) -> VlmResult:
        res: VlmResult | None = None
        if self._groq is not None:
            res = self._try_groq(self._groq, crop_bgr, shortlist, attr_schema, temperature)
        if res is None:
            res = self.ollama.verify(crop_bgr, shortlist, attr_schema, temperature)
            if not res.provider:
                res.provider = "ollama"

        # Escalate only a real-but-unsure verdict, so the extra cloud call is spent on the hard crops, not the
        # confident ones or the empty ones.
        if self._escalate is not None and res.class_name and not res.confident:
            esc = self._try_groq(self._escalate, crop_bgr, shortlist, attr_schema, temperature)
            if esc is not None and esc.class_name:
                esc.provider = "groq:escalate"
                return esc
        return res


def make_vlm_client(settings: Settings | None = None) -> VlmClient:
    """Return the vision client for the configured providers. When nothing points at the cloud, this is a
    plain OllamaVlmClient (identical to before); otherwise the RoutedVlmClient wraps cloud + local."""
    settings = settings or get_settings()
    cfg = settings.models.vlm
    uses_cloud = cfg.vision_provider == "groq" or cfg.escalate_provider == "groq"
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
                log.info("router.text_groq_fallback", error=str(exc))

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
