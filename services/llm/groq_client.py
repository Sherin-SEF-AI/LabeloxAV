"""Low-level Groq client (OpenAI-compatible chat completions), for text and vision.

Groq serves open models behind an OpenAI-shaped API, so one thin wrapper covers both the text path
(nl.py / intent) and the vision path (Path C verifier): the only difference is whether the message carries an
image part. Kept deliberately small and dependency-free (httpx, which the codebase already uses) so it slots
behind the router without pulling in an SDK.

`available()` is the gate the router asks before trying the cloud: no key means Groq is simply not
configured, and the router uses ollama instead. Every call returns parsed JSON (the callers always ask for a
strict-JSON reply) or raises, so the router can catch and fall back.
"""

from __future__ import annotations

import base64
import json

import httpx

from core.config import Settings, get_settings
from core.logging import get_logger

log = get_logger("groq")


class GroqError(RuntimeError):
    """A Groq call failed (network, auth, rate limit, or unparseable reply). The router treats any GroqError
    as a signal to fall back to the local provider."""


class GroqClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.groq

    def available(self) -> bool:
        return bool(self.cfg.api_key)

    def chat_json(self, prompt: str, *, model: str, image_jpeg: bytes | None = None,
                  temperature: float = 0.0) -> dict:
        """One chat turn that must return a JSON object. `image_jpeg`, when given, is sent as an image part
        (data URL), which is what turns this into a vision call. Raises GroqError on any failure."""
        if not self.cfg.api_key:
            raise GroqError("no GROQ_API_KEY configured")

        content: list[dict] | str
        if image_jpeg is not None:
            b64 = base64.b64encode(image_jpeg).decode()
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
        else:
            content = prompt

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(f"{self.cfg.base_url}/chat/completions", json=payload, headers=headers,
                              timeout=self.cfg.timeout_s)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return json.loads(text)
        except httpx.HTTPStatusError as exc:
            # Surface the status so a 429 (rate limit) or 401 (bad key) reads clearly in the log; the router
            # still just falls back, but an operator can see why the cloud path is being skipped.
            raise GroqError(f"groq http {exc.response.status_code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise GroqError(str(exc)[:160]) from exc
