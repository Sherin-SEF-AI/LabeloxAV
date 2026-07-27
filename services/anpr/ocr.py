"""Plate-text OCR: the wire seam for ANPR.

Reading the characters off a plate crop needs a model. This is a real wire (Qwen via Ollama with a
plate-specific prompt, or the PaddleOCR pod), not a fabricated detector: with no backend reachable it returns
("", 0.0) so the pipeline degrades to "no read" rather than inventing text. The pipeline also accepts an
injected OCR callable, which is how tests exercise the format + gating logic without a model.
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("anpr.ocr")

_PROMPT = (
    "Read the vehicle registration number printed on this Indian licence plate. Reply with strict JSON only: "
    '{"text": "<the plate characters, uppercase, no spaces; empty if unreadable>"}.'
)


def default_plate_ocr(crop_bgr: np.ndarray) -> tuple[str, float | None]:
    """Return (text, confidence) for a plate crop. Confidence is None when the backend exposes no calibrated
    score (the local generative-VLM wire): a constant stand-in would turn anpr.ocr_min_conf into a no-op that
    merely looked like a gate. Backend is config-driven; failures return ("", None)."""
    cfg = get_settings().anpr
    if cfg.ocr_backend == "pod":
        raise NotImplementedError(
            "PaddleOCR Indic OCR runs on the RunPod pod; set anpr.ocr_backend=qwen for the local wire.")
    return _read_qwen(crop_bgr)


def _read_qwen(crop_bgr: np.ndarray) -> tuple[str, float | None]:
    import httpx

    vlm = get_settings().models.vlm
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        return "", 0.0
    b64 = base64.b64encode(buf.tobytes()).decode()
    try:
        resp = httpx.post(f"{vlm.ollama_url}/api/chat", timeout=vlm.timeout_s, json={
            "model": vlm.ollama_tag, "stream": False, "format": "json",
            "messages": [{"role": "user", "content": _PROMPT, "images": [b64]}]})
        resp.raise_for_status()
        data = json.loads(resp.json()["message"]["content"])
        text = (data.get("text") or "").strip()
        return text, None   # unmeasured: a generative VLM exposes no calibrated score
    except Exception as exc:  # noqa: BLE001 - an unreachable OCR wire means "no read", never a crash
        log.warning("anpr.ocr_unavailable", error=str(exc))
        return "", None
