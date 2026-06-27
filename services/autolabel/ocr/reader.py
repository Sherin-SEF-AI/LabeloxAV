"""Road-text OCR (M2.4).

  pod path:   PaddleOCR with the Indic-script models (Devanagari, Tamil, Telugu, Kannada) for the bulk
              path, Qwen3-VL for hard/multilingual/low-confidence text.
  local path: Qwen via Ollama reads the crop (the always-available fallback).

HARD COMPLIANCE RULE: OCR never reads, stores, or indexes license-plate text. Plates are PII handled by
the anonymization gate; their bounding boxes live in PiiAudit.regions. Before OCR runs on any region, the
region is checked against the frame's plate bboxes and EXCLUDED on overlap. This is enforced here and in a
test. Stores ocr_text / ocr_lang / ocr_conf on the object and indexes the text for Phase 1 search.
"""

from __future__ import annotations

import json
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object, PiiAudit
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import crop_object

log = get_logger("ocr")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def plate_bboxes(regions: list | None) -> list:
    """Plate bounding boxes from a frame's PiiAudit.regions (the only place plate geometry lives)."""
    return [r["bbox"] for r in (regions or []) if r.get("type") == "plate" and r.get("bbox")]


def is_plate_excluded(region_bbox, plates: list, thresh: float) -> bool:
    """True if a text region overlaps any plate bbox and must NOT be OCR'd. The compliance gate."""
    return any(_iou(region_bbox, p) >= thresh for p in plates)


def read_text(crop_bgr: np.ndarray) -> tuple[str, str, float]:
    """Return (text, lang, conf). Pod: PaddleOCR Indic. Local: Qwen via Ollama."""
    cfg = get_settings().models.ocr
    if cfg.backend == "pod":
        raise NotImplementedError(
            "PaddleOCR Indic OCR runs on the RunPod pod via cloud/perception_pod.py; set "
            "models.ocr.backend=pod. Local fallback uses Qwen via Ollama.")
    return _read_qwen(crop_bgr)


def _read_qwen(crop_bgr: np.ndarray) -> tuple[str, str, float]:
    import base64

    import httpx

    vlm = get_settings().models.vlm
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        return "", "", 0.0
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = ('Read any text printed on this road sign or board. Reply with strict JSON only: '
              '{"text": "<the text, empty if none>", "lang": "<en|hi|ta|te|kn|other>"}.')
    try:
        resp = httpx.post(f"{vlm.ollama_url}/api/chat", timeout=vlm.timeout_s, json={
            "model": vlm.ollama_tag, "stream": False, "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [b64]}]})
        resp.raise_for_status()
        data = json.loads(resp.json()["message"]["content"])
        text = (data.get("text") or "").strip()
        return text, (data.get("lang") or "en"), (0.8 if text else 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("ocr.read_failed", error=str(exc))
        return "", "", 0.0


def _decode(store, uri):
    try:
        return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


async def ocr_session(session_id: UUID, limit: int | None = None) -> dict:
    """OCR text-bearing sign + board objects in a session, excluding any that overlap a plate."""
    cfg = get_settings().models.ocr
    onto = get_ontology()
    text_classes = {onto.by_name(n).id for n in ("traffic_sign", "hoarding") if onto.has_name(n)}
    store, maker = get_object_store(), get_sessionmaker()
    margin = get_settings().models.vlm.crop_margin

    async with maker() as db:
        stmt = (select(Object, Frame.img_uri, Frame.frame_id).join(Frame, Frame.frame_id == Object.frame_id)
                .where(Frame.session_id == session_id, Object.class_id.in_(text_classes), Object.state != "rejected"))
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()
        plates_by_frame: dict = {}
        for audit in (await db.execute(select(PiiAudit).where(PiiAudit.session_id == session_id))).scalars():
            plates_by_frame[audit.frame_id] = plate_bboxes(audit.regions)

    n, excluded, found = 0, 0, 0
    last_uri, last_img = None, None
    async with maker() as db:
        for obj, uri, frame_id in rows:
            plates = plates_by_frame.get(frame_id, [])
            if is_plate_excluded(list(obj.bbox), plates, cfg.plate_iou_exclude):
                o = await db.get(Object, obj.object_id)
                prov = dict(o.provenance or {})
                prov["ocr"] = {"excluded": "plate_overlap"}  # never store plate text
                o.provenance = prov
                excluded += 1
                continue
            if uri != last_uri:
                last_uri, last_img = uri, _decode(store, uri)
            if last_img is None:
                continue
            text, lang, conf = read_text(crop_object(last_img, tuple(obj.bbox), margin))
            n += 1
            if text and conf >= cfg.conf:
                o = await db.get(Object, obj.object_id)
                o.ocr_text, o.ocr_lang, o.ocr_conf = text, lang, conf
                found += 1
        await db.commit()

    out = {"session_id": str(session_id), "ocr_attempted": n, "text_found": found,
           "plate_excluded": excluded, "backend": cfg.backend}
    log.info("ocr.done", **out)
    return out
