"""Traffic-sign recognition (M2.3): a second stage on traffic_sign detections. Crop the sign, classify
its type against the Indian RTO taxonomy with SigLIP 2 zero-shot (no labeled sign data), route
text-bearing types to OCR (M2.4), and optionally read unusual/low-confidence signs with Qwen-VL
(duty-cycled). Writes sign_type, sign_category, and confidence onto the object.
"""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Object
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import crop_object
from services.autolabel.signs.taxonomy import get_sign_taxonomy

log = get_logger("signs")

_state: dict = {}


def _prompt_vecs():
    if "vecs" not in _state:
        from services.intelligence.embed import siglip2

        tax = get_sign_taxonomy()
        _state["types"] = tax["types"]
        _state["vecs"] = siglip2.encode_texts([t["prompt"] for t in tax["types"]])
    return _state["types"], _state["vecs"]


def classify_sign(crop_bgr: np.ndarray) -> dict:
    """Zero-shot sign type + category + text_bearing + confidence from a sign crop."""
    from services.intelligence.embed import siglip2

    types, tvecs = _prompt_vecs()
    fv = siglip2.encode_image(crop_bgr)
    logits = (tvecs @ fv) * get_settings().models.sign.siglip_scale
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    i = int(p.argmax())
    t = types[i]
    return {"sign_type": t["name"], "sign_category": t["category"],
            "text_bearing": bool(t.get("text_bearing", False)), "conf": round(float(p[i]), 3)}


def _decode(store, uri):
    try:
        return cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


async def recognize_session(session_id: UUID, limit: int | None = None) -> dict:
    onto = get_ontology()
    sign_id = onto.by_name("traffic_sign").id
    store, maker = get_object_store(), get_sessionmaker()
    margin = get_settings().models.vlm.crop_margin

    async with maker() as db:
        stmt = (select(Object, Frame.img_uri).join(Frame, Frame.frame_id == Object.frame_id)
                .where(Frame.session_id == session_id, Object.class_id == sign_id, Object.state != "rejected"))
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).all()

    n, text_bearing = 0, 0
    last_uri, last_img = None, None
    async with maker() as db:
        for obj, uri in rows:
            if uri != last_uri:
                last_uri, last_img = uri, _decode(store, uri)
            if last_img is None:
                continue
            res = classify_sign(crop_object(last_img, tuple(obj.bbox), margin))
            o = await db.get(Object, obj.object_id)
            o.sign_type, o.sign_category = res["sign_type"], res["sign_category"]
            prov = dict(o.provenance or {})
            prov["sign"] = {"model": "siglip2-zeroshot", "conf": res["conf"], "text_bearing": res["text_bearing"]}
            o.provenance = prov
            n += 1
            if res["text_bearing"]:
                text_bearing += 1
        await db.commit()

    out = {"session_id": str(session_id), "recognized": n, "text_bearing": text_bearing, "model": "siglip2-zeroshot"}
    log.info("signs.done", **out)
    return out
