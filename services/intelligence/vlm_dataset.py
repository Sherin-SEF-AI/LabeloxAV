"""VLM dataset generation (M-F.5, part 2): from a labeled frame plus its objects, intents, and scene-graph
relations, generate a structured multimodal training target (scene description, hazard list, per-agent intent,
and an ego-action justification). The generation is GROUNDED: the prompt carries the actual labels and the VLM
is told to describe only what they assert, and every target records the exact label ids it came from, so it is
traceable rather than free-hallucinated. A target must pass human review (status approved) before it can enter a
sellable VLM dataset; export emits only approved targets in a multimodal per-frame format.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, ObjectRelationship, Track, VlmTarget
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("intelligence.vlm_dataset")


def _region(bbox, w, h) -> str:
    cx = (bbox[0] + bbox[2]) / 2 / max(w, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(h, 1)
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else "centre"
    row = "far" if cy < 0.4 else "near"
    return f"{row}-{col}"


async def _gather(db: AsyncSession, frame_id: UUID) -> dict | None:
    onto = get_ontology()
    fr = await db.get(Frame, frame_id)
    if fr is None:
        return None
    objs = (await db.execute(select(Object).where(Object.frame_id == frame_id, Object.state != "rejected"))).scalars().all()
    w, h = float(fr.width or 1920), float(fr.height or 1080)
    obj_lines, obj_ids, track_ids = [], [], []
    for o in objs:
        try:
            nm = onto.by_id(o.class_id).name
        except Exception:  # noqa: BLE001
            nm = str(o.class_id)
        obj_lines.append(f"{nm} ({_region(o.bbox, w, h)})")
        obj_ids.append(str(o.object_id))
        if o.track_id:
            track_ids.append(o.track_id)
    # confirmed intents from the frame's tracks
    intent_lines = []
    for tid in set(track_ids):
        t = await db.get(Track, tid)
        for it in (t.intents if t else []) or []:
            if it.get("status") == "confirmed" or it.get("source") == "human":
                try:
                    cn = onto.by_id(t.class_id).name
                except Exception:  # noqa: BLE001
                    cn = "agent"
                intent_lines.append(f"{cn}: {it['intent']}")
    # confirmed scene-graph relations
    rels = (await db.execute(select(ObjectRelationship).where(
        ObjectRelationship.frame_id == frame_id, ObjectRelationship.status == "confirmed"))).scalars().all()
    id_to_name = {str(o.object_id): (onto.by_id(o.class_id).name if _safe(onto, o.class_id) else "object") for o in objs}
    rel_lines = [f"{id_to_name.get(str(r.from_object_id), 'object')} {r.kind.replace('_', ' ')} "
                 f"{id_to_name.get(str(r.to_object_id), 'object')}" for r in rels]
    return {"frame": fr, "w": w, "h": h, "objects": obj_lines, "object_ids": obj_ids,
            "track_ids": [str(t) for t in set(track_ids)], "intents": intent_lines,
            "relations": rel_lines, "relation_ids": [str(r.relationship_id) for r in rels]}


def _safe(onto, cid) -> bool:
    try:
        onto.by_id(cid)
        return True
    except Exception:  # noqa: BLE001
        return False


def _prompt(g: dict) -> str:
    return (
        "You are generating a grounded training target for an autonomous-driving vision-language dataset from a "
        "labeled Indian road frame. Use ONLY the facts in these labels; do not invent objects or events not "
        "listed.\n"
        f"Objects: {'; '.join(g['objects']) or 'none'}.\n"
        f"Agent intents: {'; '.join(g['intents']) or 'none stated'}.\n"
        f"Relations: {'; '.join(g['relations']) or 'none stated'}.\n"
        "Return STRICT JSON only: {"
        '"scene_description": "<2 sentences grounded in the objects>", '
        '"hazards": ["<hazard grounded in the labels>", ...], '
        '"agent_intents": [{"agent": "<class>", "intent": "<what it is doing>"}], '
        '"ego_action": {"action": "slow|stop|proceed|yield", "justification": "<why, from the hazards/intents>"}}.'
    )


def _ollama_generate(image_bgr, prompt: str) -> dict | None:
    import cv2
    import httpx

    cfg = get_settings().models.vlm
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return None
    payload = {"model": cfg.ollama_tag, "stream": False, "format": "json",
               "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(buf.tobytes()).decode()]}],
               "options": {"num_ctx": cfg.max_context, "temperature": 0.1}}
    try:
        resp = httpx.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=cfg.timeout_s)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm_dataset.generate_failed", error=str(exc))
        return None


async def generate_target(frame_id: UUID) -> dict:
    """Generate one grounded VLM target for a labeled frame. Stored as status=generated (awaiting review)."""
    import cv2
    import numpy as np

    from core.storage import get_object_store

    maker = get_sessionmaker()
    async with maker() as db:
        g = await _gather(db, frame_id)
        if g is None:
            return {"error": "frame not found"}
        if not g["object_ids"]:
            return {"error": "frame has no labels to ground a target on"}
        img_uri = g["frame"].img_uri
        session_id = g["frame"].session_id

    try:
        buf = np.frombuffer(get_object_store().get_bytes(img_uri), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not load frame image: {exc}"}
    content = _ollama_generate(img, _prompt(g))
    if content is None:
        return {"error": "VLM generation failed (is Ollama up?)"}

    async with maker() as db:
        tgt = VlmTarget(frame_id=frame_id, session_id=session_id, kind="scene_pack", content=content,
                        grounding={"object_ids": g["object_ids"], "track_ids": g["track_ids"],
                                   "relation_ids": g["relation_ids"]},
                        model=get_settings().models.vlm.ollama_tag, status="generated")
        db.add(tgt)
        await db.commit()
        tid = tgt.target_id
    log.info("vlm_dataset.generated", frame_id=str(frame_id), target_id=str(tid))
    return {"target_id": str(tid), "content": content,
            "grounding": {"objects": len(g["object_ids"]), "intents": len(g["intents"]), "relations": len(g["relation_ids"])}}


async def set_target_status(target_id: UUID, status: str) -> dict:
    """Human review gate: approve or reject a generated target. Only approved targets export."""
    if status not in ("approved", "rejected", "generated"):
        return {"error": "status must be approved | rejected | generated"}
    maker = get_sessionmaker()
    async with maker() as db:
        t = await db.get(VlmTarget, target_id)
        if t is None:
            return {"error": "target not found"}
        t.status = status
        await db.commit()
        return {"target_id": str(target_id), "status": status}


async def list_targets(frame_id: UUID) -> list[dict]:
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(select(VlmTarget).where(VlmTarget.frame_id == frame_id))).scalars().all()
        return [{"target_id": str(t.target_id), "kind": t.kind, "content": t.content, "grounding": t.grounding,
                 "status": t.status, "model": t.model} for t in rows]


async def export_dataset(session_id: UUID | None = None) -> dict:
    """Export the APPROVED VLM targets in a multimodal per-frame format (image reference + grounded targets)."""
    maker = get_sessionmaker()
    async with maker() as db:
        q = select(VlmTarget, Frame.img_uri).join(Frame, Frame.frame_id == VlmTarget.frame_id).where(
            VlmTarget.status == "approved")
        if session_id is not None:
            q = q.where(VlmTarget.session_id == session_id)
        rows = (await db.execute(q)).all()
    samples = [{"frame_id": str(t.frame_id), "image": img_uri, "target": t.content, "grounding": t.grounding,
                "model": t.model} for t, img_uri in rows]
    return {"format": "labelox-vlm-multimodal-v1", "n_samples": len(samples), "samples": samples}
