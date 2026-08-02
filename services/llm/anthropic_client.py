"""Low-level Anthropic client (Messages API), for text and vision.

Groq already covers the cloud path, so the reason to add another is not redundancy. Groq serves open models,
and the job this provider exists for is judging: deciding whether an existing label is right, on a 48px crop
of an Indian street, where the alternatives are an autorickshaw, an e-rickshaw and a tempo. That is the task
the corpus has been unable to get an answer to at any scale, and it is worth a stronger model than the one
chosen for throughput on the autolabel path.

Shaped like `GroqClient` on purpose: same `available()` gate, same "returns parsed JSON or raises" contract,
same httpx-only dependency rather than an SDK. The router then treats a frontier provider exactly as it
treats Groq, including the circuit breaker and the fall back to local, so the cloud is never a hard
dependency and no new failure mode enters the system.

The one shape difference the Messages API forces: images are a `source` block with base64 and an explicit
media type rather than an OpenAI-style data URL, and there is no `response_format`, so strict JSON is
requested in the prompt and the reply is unwrapped from any prose fence before parsing.
"""

from __future__ import annotations

import base64
import json
import re

import httpx

from core.config import Settings, get_settings
from core.logging import get_logger

log = get_logger("anthropic")

# Claude replies to a "JSON only" instruction with bare JSON, but a fenced block is a documented failure mode
# under temperature, and losing an otherwise good verdict to a pair of backticks would be a silly way to
# waste a judged crop.
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S)


class AnthropicError(RuntimeError):
    """An Anthropic call failed (network, auth, rate limit, or unparseable reply). The router treats any
    AnthropicError as a signal to fall back, exactly as it treats GroqError."""


def _unwrap_json(text: str) -> dict:
    m = _FENCE.search(text)
    return json.loads(m.group(1) if m else text)


class AnthropicClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.anthropic

    def available(self) -> bool:
        return bool(self.cfg.api_key)

    def chat_json(self, prompt: str, *, model: str, image_jpeg: bytes | None = None,
                  temperature: float = 0.0) -> dict:
        """One turn that must return a JSON object. `image_jpeg`, when given, makes it a vision call.

        Raises AnthropicError on any failure, so the router can fall back without inspecting the cause.
        """
        if not self.cfg.api_key:
            raise AnthropicError("no ANTHROPIC_API_KEY configured")

        content: list[dict] = []
        if image_jpeg is not None:
            # Image before text: the model attends to the instruction with the picture already in context,
            # which is the ordering Anthropic documents for vision prompts.
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(image_jpeg).decode()}})
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {"x-api-key": self.cfg.api_key, "anthropic-version": self.cfg.api_version,
                   "content-type": "application/json"}
        try:
            resp = httpx.post(f"{self.cfg.base_url}/messages", json=payload, headers=headers,
                              timeout=self.cfg.timeout_s)
            resp.raise_for_status()
            blocks = resp.json().get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if not text.strip():
                raise AnthropicError("empty reply")
            return _unwrap_json(text)
        except httpx.HTTPStatusError as exc:
            # Surface the status so a 429 (rate limit) or 401 (bad key) reads clearly in the log. The router
            # only falls back, but an operator needs to see why the cloud path is being skipped, and a
            # silently-degraded judge is worse than no judge.
            raise AnthropicError(f"anthropic http {exc.response.status_code}") from exc
        except AnthropicError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AnthropicError(str(exc)[:160]) from exc
