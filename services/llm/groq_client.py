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
import time

import cv2
import httpx
import numpy as np

from core.config import Settings, get_settings
from core.logging import get_logger

log = get_logger("groq")


# Groq refuses any image under 32 pixels on a side ("Image must have at least 32 pixels in each dimension").
# Distant objects make crops that small all the time: measured against 260 calibration crops, 21 were refused,
# and in the autolabel stream each refusal fell back to the local model, which is the load a cloud provider
# is configured to take away.
MIN_SIDE_PX = 32

# A rate limit on Groq clears in well under a second: every refusal seen during calibration named a wait of
# 120 to 440 ms. Waiting that long once is far cheaper than the fallback, which hands the call to the local
# model on the same machine the cloud is meant to spare. Longer waits are not worth holding a worker for.
MAX_RETRY_WAIT_S = 5.0
RATE_LIMIT_RETRIES = 2


def ensure_min_side(image_jpeg: bytes, min_side: int = MIN_SIDE_PX) -> bytes:
    """Enlarge a JPEG so both sides are at least `min_side`, keeping its aspect ratio. Larger images pass
    through untouched, byte for byte."""
    img = cv2.imdecode(np.frombuffer(image_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return image_jpeg
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return image_jpeg
    scale = min_side / max(1, min(h, w))
    img = cv2.resize(img, (max(min_side, round(w * scale)), max(min_side, round(h * scale))),
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes() if ok else image_jpeg


def _retry_after_s(resp: httpx.Response) -> float | None:
    """How long Groq asked us to wait, from the retry-after header or the "try again in Nms" message."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    import re

    m = re.search(r"try again in ([0-9.]+)(ms|s)", resp.text)
    if m:
        return float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
    return None


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
            b64 = base64.b64encode(ensure_min_side(image_jpeg)).decode()
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
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = httpx.post(f"{self.cfg.base_url}/chat/completions", json=payload, headers=headers,
                                  timeout=self.cfg.timeout_s)
            except Exception as exc:  # noqa: BLE001
                raise GroqError(str(exc)[:160]) from exc
            # Groq reports a rate limit as 429, and sometimes as a 400 whose code is rate_limit_exceeded.
            limited = resp.status_code == 429 or (resp.status_code == 400 and "rate_limit_exceeded" in resp.text)
            wait = _retry_after_s(resp) if limited else None
            if limited and attempt < RATE_LIMIT_RETRIES and wait is not None and wait <= MAX_RETRY_WAIT_S:
                time.sleep(wait + 0.05)
                continue
            if resp.status_code >= 400:
                # Surface the status and Groq's own message, so a refusal reads clearly in the log. The router
                # still just falls back, but an operator can see why the cloud path is being skipped.
                raise GroqError(f"groq http {resp.status_code}: {resp.text[:160]}")
            try:
                text = resp.json()["choices"][0]["message"]["content"]
                return json.loads(text)
            except Exception as exc:  # noqa: BLE001
                raise GroqError(f"unparseable reply: {str(exc)[:120]}") from exc
        raise GroqError("groq rate limited after retries")
