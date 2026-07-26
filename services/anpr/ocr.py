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


def default_plate_ocr(crop_bgr: np.ndarray) -> tuple[str, float]:
    """Return (text, confidence) for a plate crop. Backend is config-driven; failures return ("", 0.0)."""
    cfg = get_settings().anpr
    if cfg.ocr_backend == "pod":
        raise NotImplementedError(
            "PaddleOCR Indic OCR runs on the RunPod pod; set anpr.ocr_backend=qwen for the local wire.")
    return _read_qwen(crop_bgr)


def _read_qwen(crop_bgr: np.ndarray) -> tuple[str, float]:
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
        return text, (0.8 if text else 0.0)
    except Exception as exc:  # noqa: BLE001 - an unreachable OCR wire means "no read", never a crash
        log.warning("anpr.ocr_unavailable", error=str(exc))
        return "", 0.0
