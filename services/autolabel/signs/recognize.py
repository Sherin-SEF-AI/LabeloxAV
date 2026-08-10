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
    """Type and negative prompt vectors, embedded once.

    Cached in a module global keyed on the taxonomy version, so editing the YAML in a long-running process
    rebuilds the vectors instead of silently serving prompts that no longer exist on disk.
    """
    tax = get_sign_taxonomy()
    if _state.get("version") != tax["version"]:
        from services.intelligence.embed import siglip2

        _state["version"] = tax["version"]
        _state["types"] = tax["types"]
        _state["negatives"] = tax["negatives"]
        _state["vecs"] = siglip2.encode_texts([t["prompt"] for t in tax["types"]])
        _state["neg_vecs"] = (siglip2.encode_texts([n["prompt"] for n in tax["negatives"]])
                              if tax["negatives"] else None)
    return _state["types"], _state["vecs"], _state["neg_vecs"]


def _similarities(crop_bgr: np.ndarray):
    """Cosine similarity of the crop against every type prompt and every negative prompt."""
    from services.intelligence.embed import siglip2

    types, tvecs, nvecs = _prompt_vecs()
    fv = siglip2.encode_image(crop_bgr)
    return types, tvecs @ fv, (nvecs @ fv if nvecs is not None else None)


def sign_margin(crop_bgr: np.ndarray) -> float:
    """How much more this crop looks like a sign than like the things that are not signs.

    Positive means the best sign prompt beat every negative. This is the quantity the decision rests on, so
    it is what gets reported rather than a softmax probability, which exists for any input at all and so
    measures nothing.
    """
    _types, sims, nsims = _similarities(crop_bgr)
    if nsims is None or not len(nsims):
        return float(sims.max())
    return float(sims.max() - nsims.max())


def classify_sign(crop_bgr: np.ndarray) -> dict:
    """The sign's type, or an explicit refusal to type it.

    Measured on real corpus crops, the previous version scored a photograph of a bus at 0.817 as `bus_stop`
    where genuine signs averaged 0.759: the prompt "a bus stop information sign" matches a bus, because the
    object noun carries the phrase. Softmaxing 21 sign prompts could not express "this is not a sign", so
    every crop handed in came back confidently typed, including crops of vehicles, people and blank sky.

    Now the best type competes against prompts for what a sign is not, and has to win by a margin. Below
    that margin `sign_type` is None, which is a smaller claim and a true one.
    """
    cfg = get_settings().models.sign
    types, sims, nsims = _similarities(crop_bgr)
    i = int(sims.argmax())
    t = types[i]
    best_neg = float(nsims.max()) if nsims is not None and len(nsims) else float("-inf")
    margin = float(sims[i]) - best_neg if best_neg != float("-inf") else float(sims[i])

    out = {"sign_type": t["name"], "sign_category": t["category"],
           "text_bearing": bool(t.get("text_bearing", False)),
           "margin": round(margin, 4), "top_similarity": round(float(sims[i]), 4),
           "rejected": False, "reason": None}

    if margin < cfg.min_margin:
        # Named rather than left as a bare None, because "which negative won" is what tells a reviewer
        # whether the detector boxed a hoarding, a vehicle, or the back of a sign.
        worst = _state["negatives"][int(nsims.argmax())]["name"] if nsims is not None and len(nsims) else None
        out.update(sign_type=None, sign_category=None, text_bearing=False, rejected=True,
                   reason=f"looks more like {worst} than any sign type" if worst
                          else "no sign type scored high enough to name")
    return out


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

    n, text_bearing, rejected = 0, 0, 0
    last_uri, last_img = None, None
    async with maker() as db:
        for obj, uri in rows:
            if uri != last_uri:
                last_uri, last_img = uri, _decode(store, uri)
            if last_img is None:
                continue
            res = classify_sign(crop_object(last_img, tuple(obj.bbox), margin))
            o = await db.get(Object, obj.object_id)
            # A rejection is written, not skipped. Clearing the columns is how a re-run corrects a type this
            # object was given by the previous version, which typed everything it was handed.
            o.sign_type, o.sign_category = res["sign_type"], res["sign_category"]
            prov = dict(o.provenance or {})
            prov["sign"] = {"model": "siglip2-zeroshot+negatives", "margin": res["margin"],
                            "top_similarity": res["top_similarity"], "text_bearing": res["text_bearing"],
                            "rejected": res["rejected"], "reason": res["reason"]}
            o.provenance = prov
            if res["rejected"]:
                rejected += 1
            else:
                n += 1
                if res["text_bearing"]:
                    text_bearing += 1
        await db.commit()

    out = {"session_id": str(session_id), "examined": len(rows), "recognized": n,
           "rejected": rejected, "text_bearing": text_bearing, "model": "siglip2-zeroshot+negatives"}
    log.info("signs.done", **out)
    return out
