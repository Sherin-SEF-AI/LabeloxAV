"""Multi-model reconciliation for the uncertain tail. When the primary detectors disagree (no cross-path
agreement) or the consistency critic flags a class/relationship problem, that disagreement is the signal to
ask an INDEPENDENT model. SigLIP 2 zero-shot (classify_crop) is a different model family from the
YOLO/YOLO-World detectors, so its read on the crop is a real second opinion, not an echo. The verdict:

  - confirm : the independent model's top class is the current class -> corroboration.
  - correct : it strongly prefers a different class -> a suggested relabel for a human (or, only when the
              caller opts in, an agent-applied correction).
  - unsure  : it is not confident enough to adjudicate -> stays for a human.

This is read-only: it returns opinions and suggestions, it never mutates labels. The structure (a verdict
plus alternatives per object) is model-agnostic, so a heavier VLM reasoner can replace classify_crop for
the hardest cases without changing callers.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("agent.reconcile")


def reconcile_crop(crop_bgr, current_class_id: int, *, min_conf: float = 0.40) -> dict:
    """Adjudicate one crop with the independent model. Returns a verdict + the top alternatives."""
    from services.autolabel.classify_crop import classify_crop

    preds = classify_crop(crop_bgr, topk=3)
    if not preds:
        return {"verdict": "unsure", "conf": 0.0, "alternatives": []}
    top = preds[0]
    if int(top["class_id"]) == int(current_class_id):
        return {"verdict": "confirm", "suggested_class_id": current_class_id,
                "suggested_class_name": top["class_name"], "conf": round(float(top["conf"]), 3),
                "alternatives": preds}
    if float(top["conf"]) >= min_conf:
        return {"verdict": "correct", "suggested_class_id": int(top["class_id"]),
                "suggested_class_name": top["class_name"], "conf": round(float(top["conf"]), 3),
                "alternatives": preds}
    return {"verdict": "unsure", "conf": round(float(top["conf"]), 3), "alternatives": preds}


async def reconcile_frame(db: AsyncSession, frame_id: uuid.UUID, object_ids: list[str] | None = None) -> dict:
    """Reconcile a frame's objects (or a given subset) against the independent model. Read-only."""
    from core.storage import get_object_store
    from services.autolabel.ontology import get_ontology
    from services.recall.backends import load_image_bgr

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    q = select(Object).where(Object.frame_id == frame_id, Object.source != "human")
    if object_ids:
        q = q.where(Object.object_id.in_([uuid.UUID(o) for o in object_ids]))
    objs = list((await db.execute(q)).scalars().all())
    if not objs:
        return {"frame_id": str(frame_id), "reconciled": 0, "items": []}

    img = load_image_bgr(get_object_store(), frame.img_uri)
    h, w = img.shape[:2]
    onto = get_ontology()
    items: list[dict] = []
    tally = {"confirm": 0, "correct": 0, "unsure": 0}
    for o in objs:
        x1, y1, x2, y2 = (int(round(float(v))) for v in o.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = img[y1:y2, x1:x2]
        r = reconcile_crop(crop, int(o.class_id))
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        try:
            cur = onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            cur = str(o.class_id)
        items.append({"object_id": str(o.object_id), "current_class": cur, **r})
    log.info("agent.reconcile", frame_id=str(frame_id), **tally)
    return {"frame_id": str(frame_id), "reconciled": len(items), "verdicts": tally, "items": items}
