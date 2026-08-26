"""Natural-language and VLM-assisted bulk editing (M-F.3): the annotator types an operation over the current
frame or session ("select every parked autorickshaw", "reclassify the fallback objects to push_cart", "flag
every pedestrian with no visible mask"). The command is parsed into a selection predicate and an operation,
resolved to concrete objects, and PREVIEWED. Nothing changes until the human confirms. On confirmation the edit
lands as one reversible AgentRun with per-object provenance, routes the touched objects to review, and writes an
audit entry. Agents (and language) propose; the gate disposes.

The selection uses class and attribute filters, a missing-mask predicate, and, for referential phrases the
structure cannot resolve ("near the barrier"), an optional duty-cycled VLM confirmation per candidate.
"""

from __future__ import annotations

import re
import uuid
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object
from services.agent.class_move import refuse_reason
from services.agent.nl import _resolve_classes
from services.autolabel.ontology import get_ontology

log = get_logger("agent.nl_edit")

# attribute keywords -> the attrs values (or intent) they match. Kept small and explicit.
_ATTR_WORDS = {
    "parked": ("motion", {"parked", "static", "stopped", "stationary"}),
    "moving": ("motion", {"moving", "driving"}),
    "occluded": ("occlusion", {"partial", "heavy", "full", "occluded"}),
    "braking": ("brake", {"braking", "on"}),
}
_REFERENTIAL = re.compile(r"\bnear\b|\bnext to\b|\bbeside\b|\bbehind\b|\bin front of\b|\bunder\b|\bon the\b", re.I)


def parse_edit(text: str, onto) -> dict:
    """Parse a command into {operation, select_class_ids, to_class_id, mask_missing, attr, referential, raw}."""
    t = text.strip().lower()

    # operation
    to_class_id = None
    if re.search(r"\breclassify|relabel|change .* to |set .* to \b", t) or re.search(r"\bto\s+[a-z_]+\s*$", t):
        operation = "reclassify"
        m = re.search(r"\bto\s+([a-z][a-z0-9_ ]*)\s*$", t)
        if m:
            cand = m.group(1).strip().replace(" ", "_")
            if onto.has_name(cand):
                to_class_id = onto.by_name(cand).id
    elif re.search(r"\bflag\b|\bmark for review\b|\bsend .* to review\b", t):
        operation = "flag"
    else:
        operation = "select"

    # the SELECTION classes: strip the "to <class>" target so it is not also treated as a selector
    sel_text = re.sub(r"\bto\s+[a-z][a-z0-9_ ]*\s*$", "", t)
    names = _resolve_classes(sel_text, onto)
    select_ids: set[int] = set()
    if "fallback" in sel_text:
        select_ids.update(onto.fallback_ids())
    for n in names:
        try:
            select_ids.add(onto.by_name(n).id)
        except Exception:  # noqa: BLE001
            pass

    mask_missing = bool(re.search(r"\bno (visible )?(box )?mask\b|\bwithout (a )?mask\b|\bmissing mask\b", t))
    attr = None
    for word, spec in _ATTR_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            attr = {"name": spec[0], "values": sorted(spec[1])}
            break

    return {"operation": operation, "select_class_ids": sorted(select_ids), "to_class_id": to_class_id,
            "mask_missing": mask_missing, "attr": attr, "referential": bool(_REFERENTIAL.search(t)),
            "raw": text.strip()}


def _matches_attr(obj: Object, attr: dict | None) -> bool:
    if not attr:
        return True
    vals = {str(v).lower() for v in (obj.attrs or {}).values()}
    return bool(vals & set(attr["values"])) or str((obj.attrs or {}).get(attr["name"], "")).lower() in attr["values"]


async def resolve(db: AsyncSession, plan: dict, *, frame_id: UUID | None = None, session_id: UUID | None = None,
                  limit: int = 300) -> dict:
    """Resolve a parsed plan to concrete objects (no mutation). Returns the preview list + count + warnings."""
    onto = get_ontology()
    q = select(Object, Frame.frame_id, Frame.session_id).join(Frame, Frame.frame_id == Object.frame_id)
    if frame_id is not None:
        q = q.where(Object.frame_id == frame_id)
    elif session_id is not None:
        q = q.where(Frame.session_id == session_id)
    q = q.where(Object.state != "rejected")
    if plan["select_class_ids"]:
        q = q.where(Object.class_id.in_(plan["select_class_ids"]))
    if plan["mask_missing"]:
        q = q.where(Object.mask_uri.is_(None))
    rows = (await db.execute(q.limit(limit * 4))).all()

    matched = []
    for obj, fid, sid in rows:
        if not _matches_attr(obj, plan["attr"]):
            continue
        matched.append({"object_id": str(obj.object_id), "frame_id": str(fid), "class_id": obj.class_id,
                        "class_name": onto.by_id(obj.class_id).name, "bbox": [float(x) for x in obj.bbox],
                        "state": obj.state, "crop_url": f"/api/objects/{obj.object_id}/crop"})
        if len(matched) >= limit:
            break

    warnings = []
    if not plan["select_class_ids"] and not plan["mask_missing"] and not plan["attr"]:
        warnings.append("no class, attribute, or mask filter recognized; this would match everything in scope")
    if plan["operation"] == "reclassify" and plan["to_class_id"] is None:
        warnings.append("reclassify needs a valid target class after 'to'; none recognized")
    if plan["referential"]:
        warnings.append("this command references spatial context; enable VLM refine to confirm each candidate")
    return {"plan": plan, "count": len(matched), "objects": matched, "warnings": warnings}


async def vlm_refine(matched: list[dict], phrase: str, max_objs: int = 40) -> list[str]:
    """Keep the objects a VLM agrees match a referential phrase. Duty-cycled; returns kept object_ids."""
    import cv2
    import numpy as np

    from core.storage import get_object_store
    from db.session import get_sessionmaker
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient, crop_object

    vlm = OllamaVlmClient()
    kept: list[str] = []
    maker = get_sessionmaker()
    async with maker() as db:
        for m in matched[:max_objs]:
            obj = await db.get(Object, UUID(m["object_id"]))
            fr = await db.get(Frame, UUID(m["frame_id"]))
            if obj is None or fr is None:
                continue
            try:
                buf = np.frombuffer(get_object_store().get_bytes(fr.img_uri), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                res = vlm.verify(crop_object(img, tuple(float(x) for x in obj.bbox), 0.4),
                                 [f"yes ({phrase})", "no"], {})
                if (res.class_name or "").lower().startswith("yes"):
                    kept.append(m["object_id"])
            except Exception as exc:  # noqa: BLE001
                log.info("nl_edit.vlm_refine_failed", error=str(exc))
    return kept


async def apply_edit(db: AsyncSession, plan: dict, object_ids: list[UUID], *, created_by: str | None = None) -> dict:
    """Apply a bulk edit to the confirmed objects as ONE reversible AgentRun. reclassify sets the class and
    routes to review; flag routes to review with a flag. Every object is stamped with the run and audited."""
    onto = get_ontology()
    op = plan["operation"]
    if op == "reclassify" and plan["to_class_id"] is None:
        return {"error": "reclassify needs a valid target class"}
    if op == "select":
        return {"error": "select is a preview only; nothing to apply"}

    run_id = uuid.uuid4()
    objs = (await db.execute(select(Object).where(Object.object_id.in_(object_ids)))).scalars().all()
    changes: dict = {}
    skipped: dict[str, int] = {}
    for obj in objs:
        # A language command must not overwrite a human's label. Every other bulk write path excludes
        # source == "human"; this one did not, so one sentence could undo a person's work.
        if obj.source == "human":
            skipped["human_label"] = skipped.get("human_label", 0) + 1
            continue
        # And it may sharpen a class, not change what kind of thing it is - the same l0 boundary the
        # relabel and contamination paths already refuse to cross. A typed sentence is if anything a
        # looser filter than a detector's confidence, so it needs the guard more, not less.
        if op == "reclassify":
            refusal = refuse_reason(onto, int(obj.class_id), int(plan["to_class_id"]))
            if refusal:
                skipped["category_change"] = skipped.get("category_change", 0) + 1
                log.warning("agent.nl_edit.refused_category_change", object_id=str(obj.object_id),
                            reason=refusal, command=plan["raw"])
                continue
        changes[str(obj.object_id)] = {"from_class": int(obj.class_id), "from_state": obj.state,
                                       "from_source": obj.source}
        if op == "reclassify":
            obj.class_id = int(plan["to_class_id"])
        obj.state = "review"                    # a human confirms every language-issued edit
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("nl_edit", []).append({"op": op, "command": plan["raw"],
                                               "to": onto.by_id(int(plan["to_class_id"])).name if plan["to_class_id"] else None})
        obj.provenance = prov

    db.add(AgentRun(run_id=run_id, kind="nl_edit", scope={"command": plan["raw"], "operation": op},
                    status="committed", policy=plan, counts={"edited": len(changes)}, changes=changes,
                    critic={}, created_by=created_by))
    from services.govern.audit import record

    await record(db, actor="nl_edit", decision=f"bulk_{op}", subject=str(run_id),
                 rationale={"command": plan["raw"], "n_objects": len(changes),
                            "to_class": onto.by_id(int(plan["to_class_id"])).name if plan["to_class_id"] else None},
                 commit=False)
    await db.commit()
    log.info("nl_edit.applied", run_id=str(run_id), op=op, edited=len(changes))
    if skipped:
        # Reported rather than silent: a command that matched fifty objects and edited thirty has to say
        # what happened to the other twenty, or the preview count and the result disagree with no reason.
        log.info("nl_edit.skipped", run_id=str(run_id), **skipped)
    return {"run_id": str(run_id), "operation": op, "edited": len(changes), "routed_to": "review",
            "skipped": skipped}
