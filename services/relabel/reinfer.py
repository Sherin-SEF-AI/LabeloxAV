"""Bulk re-inference proposals (M4.2). Two proposal sources:

  ontology promotion - when a fallback class is promoted to a named class (vehicle_fallback -> water_tanker
                       once that class exists), re-classify the affected objects. Deterministic, local.
  model re-inference - the champion model re-infers the frames on the A100; the pod emits relabeled.jsonl
                       which parse_model_proposals turns into the same proposal shape.

A proposal records the old and new label so the diff and provenance are a single walk. The heavy model
pass is the A100 relabel burst (services/relabel/cloud.py); ontology promotion runs locally.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from services.autolabel.ontology import get_ontology

log = get_logger("relabel_reinfer")


async def propose_ontology_promotion(db: AsyncSession, from_class: str, to_class: str,
                                     session_ids: list[str] | None = None) -> list[dict]:
    """Promote a scoped set of from_class objects to to_class (a named refinement of a fallback). The class
    label is refined; the detection confidence is unchanged. Human-verified objects are still surfaced as
    proposals so the apply step can route them to review rather than silently overwriting them."""
    onto = get_ontology()
    if not onto.has_name(from_class) or not onto.has_name(to_class):
        return []
    from_id, to_id = onto.by_name(from_class).id, onto.by_name(to_class).id

    q = select(Object.object_id, Object.class_id, Object.conf, Object.source, Object.state).where(Object.class_id == from_id)
    if session_ids:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id.in_(session_ids))
    rows = (await db.execute(q)).all()
    proposals = [{"object_id": str(oid), "old_class_id": from_id, "old_class": from_class, "old_conf": float(conf or 0.0),
                  "new_class_id": to_id, "new_class": to_class, "new_conf": float(conf or 0.0),
                  "source": src, "state": st, "reason": "ontology_promotion"}
                 for oid, _cid, conf, src, st in rows]
    log.info("relabel.propose_promotion", from_class=from_class, to_class=to_class, n=len(proposals))
    return proposals


async def parse_model_proposals(db: AsyncSession, model_output: list[dict]) -> list[dict]:
    """Turn champion-model re-inference output ([{object_id, class_name, conf}]) into proposals, joining the
    existing label so the diff has both sides."""
    onto = get_ontology()
    by_id = {str(r[0]): r for r in (await db.execute(
        select(Object.object_id, Object.class_id, Object.conf, Object.source, Object.state).where(
            Object.object_id.in_([m["object_id"] for m in model_output]))) ).all()}
    out = []
    for m in model_output:
        cur = by_id.get(str(m["object_id"]))
        if cur is None or not onto.has_name(m["class_name"]):
            continue
        _oid, old_cid, old_conf, src, st = cur
        out.append({"object_id": str(m["object_id"]), "old_class_id": old_cid, "old_class": onto.by_id(old_cid).name,
                    "old_conf": float(old_conf or 0.0), "new_class_id": onto.by_name(m["class_name"]).id,
                    "new_class": m["class_name"], "new_conf": float(m.get("conf", 0.0)),
                    "source": src, "state": st, "reason": "model_reinfer"})
    return out
