"""Conversational corpus copilot: ask the dataset a question in plain language and get the matching frames.
Where the per-frame NL control (services/agent/nl.py) drives agent actions on one frame, this answers
multi-facet questions across the whole corpus -- "pedestrians crossing against traffic at night",
"near-misses with a two-wheeler on the highway" -- by parsing the sentence into scene, object-class,
attribute, and safety facets and composing them into one query. It reuses the class resolver from nl.py and
degrades to whatever facets it recognized; the parsed facets are returned so the person sees exactly how the
question was understood.
"""

from __future__ import annotations

import re

from sqlalchemy import Integer, and_, distinct, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectDynamics

log = get_logger("agent.copilot")

# scene phrase -> (axis, value)
_SCENE = {
    "night": ("time_of_day", "night"), "day": ("time_of_day", "day"), "dusk": ("time_of_day", "dusk"),
    "dawn": ("time_of_day", "dawn"), "rain": ("weather", "rain"), "rainy": ("weather", "rain"),
    "fog": ("weather", "fog"), "foggy": ("weather", "fog"), "overcast": ("weather", "overcast"),
    "clear": ("weather", "clear"), "highway": ("road_type", "highway"), "urban": ("road_type", "urban"),
    "residential": ("road_type", "residential"), "rural": ("road_type", "rural"),
    "dense": ("density", "dense"), "heavy traffic": ("density", "dense"), "sparse": ("density", "sparse"),
}
# attribute phrase -> (attr, matcher) where matcher is a value, True/False, or ">0"
_ATTR = {
    "wrong way": ("direction", "wrong_way"), "wrong-way": ("direction", "wrong_way"),
    "against traffic": ("direction", "wrong_way"), "oncoming": ("direction", "wrong_way"),
    "crossing": ("direction", "cross"), "same direction": ("direction", "same"),
    "no helmet": ("helmet", False), "without helmet": ("helmet", False), "helmetless": ("helmet", False),
    "occluded": ("occlusion", ">0"), "static": ("static", True), "parked": ("static", True),
    "stationary": ("static", True), "moving": ("static", False),
}


def parse_query(text: str, onto) -> dict:
    from services.agent.nl import _resolve_classes

    t = text.lower().strip()
    scene = {}
    for phrase, (axis, val) in _SCENE.items():
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            scene[axis] = val
    attrs = {}
    for phrase, (k, v) in _ATTR.items():
        if phrase in t:
            attrs[k] = v
    class_names = _resolve_classes(t, onto)
    class_ids = {c.id for c in onto.classes if c.name in class_names}
    safety = bool(re.search(r"\bnear[- ]?miss(es)?\b|\bclose call\b|\balmost hit\b", t))
    return {"scene": scene, "attrs": attrs, "classes": sorted(class_names), "class_ids": class_ids, "safety": safety}


async def answer_corpus_query(db: AsyncSession, text: str, *, limit: int = 40) -> dict:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    facets = parse_query(text, onto)

    q = select(distinct(Frame.frame_id), Frame.session_id)
    for axis, val in facets["scene"].items():
        q = q.where(Frame.scene[axis].astext == val)

    needs_object = bool(facets["class_ids"] or facets["attrs"] or facets["safety"])
    if needs_object:
        # a matching object must exist on the frame (class + attrs), and for safety a low-TTC one
        oconds = [Object.frame_id == Frame.frame_id, Object.source != "human"]
        if facets["class_ids"]:
            oconds.append(Object.class_id.in_(list(facets["class_ids"])))
        for k, v in facets["attrs"].items():
            if v == ">0":
                oconds.append(Object.attrs[k].astext.cast(Integer) > 0)
            elif isinstance(v, bool):
                oconds.append(Object.attrs[k].astext == ("true" if v else "false"))
            else:
                oconds.append(Object.attrs[k].astext == v)
        obj_exists = exists(select(Object.object_id).where(and_(*oconds)))
        q = q.where(obj_exists)
        if facets["safety"]:
            safe_exists = exists(select(ObjectDynamics.object_id).join(Object, Object.object_id == ObjectDynamics.object_id)
                                 .where(Object.frame_id == Frame.frame_id, ObjectDynamics.ttc_s.isnot(None), ObjectDynamics.ttc_s < 2.5))
            q = q.where(safe_exists)

    rows = (await db.execute(q.limit(limit))).all()
    frames = [{"frame_id": str(fid), "session_id": str(sid)} for fid, sid in rows]
    parts = []
    if facets["classes"]:
        parts.append(", ".join(facets["classes"][:4]) + (" objects" if len(facets["classes"]) == 1 else ""))
    if facets["attrs"]:
        parts.append(", ".join(f"{k}={v}" for k, v in facets["attrs"].items()))
    if facets["scene"]:
        parts.append(", ".join(f"{a}={v}" for a, v in facets["scene"].items()))
    if facets["safety"]:
        parts.append("near-miss (TTC < 2.5s)")
    understood = " · ".join(parts) or "everything"
    log.info("agent.copilot", results=len(frames), understood=understood)
    return {"understood": understood, "facets": {k: facets[k] for k in ("scene", "attrs", "classes", "safety")},
            "count": len(frames), "frames": frames}


async def suggest_for_frame(db: AsyncSession, frame_id) -> dict:
    """Proactive in-editor assistant: look at the current frame and offer the agent actions that would help,
    each with the count it would touch, so the annotator sees the highest-leverage next step at a glance."""
    import uuid as _uuid

    from services.agent.attribute_agent import plan_attributes
    from services.agent.cuboid_agent import plan_cuboids
    from services.agent.critic import critique_frame
    from services.agent.frame_agent import _build_context, _load_objects

    fid = frame_id if isinstance(frame_id, _uuid.UUID) else _uuid.UUID(str(frame_id))
    suggestions: list[dict] = []
    try:
        cub = await plan_cuboids(db, fid)
        n = cub["counts"]["auto_accept"] + cub["counts"]["review"]
        if n:
            suggestions.append({"action": "fit_cuboids", "label": f"fit {n} 3D box{'es' if n != 1 else ''} from the 2D boxes", "n": n, "score": 0.6})
    except Exception:  # noqa: BLE001
        pass
    try:
        att = await plan_attributes(db, fid)
        if att["counts"]["attrs_filled"]:
            suggestions.append({"action": "fill_attributes", "label": f"auto-fill {att['counts']['attrs_filled']} attributes on {att['counts']['objects']} objects", "n": att["counts"]["attrs_filled"], "score": 0.5})
    except Exception:  # noqa: BLE001
        pass
    try:
        from db.models import Frame
        frame = await db.get(Frame, fid)
        objs = await _load_objects(db, fid)
        if frame and objs:
            ctx = await _build_context(db, frame, objs)
            flagged = sum(1 for v in critique_frame(ctx).values() if not v.ok)
            if flagged:
                suggestions.append({"action": "review_flagged", "label": f"{flagged} object{'s' if flagged != 1 else ''} look inconsistent (critic flag)", "n": flagged, "score": 0.8})
    except Exception:  # noqa: BLE001
        pass
    suggestions.sort(key=lambda s: -s["score"])
    return {"frame_id": str(fid), "suggestions": suggestions}


async def dataset_report(db: AsyncSession) -> dict:
    """Auto dataset report: one shareable snapshot of corpus health -- size, class balance, coverage gaps,
    the fix-queue error summary, and mined safety/rare scenarios."""
    from sqlalchemy import func, select

    from db.models import ErrorCandidate, Object, ScenarioCandidate
    from db.models import Session as DbSession
    from services.agent.coverage import analyze_coverage

    coverage = await analyze_coverage(db)
    n_sessions = (await db.execute(select(func.count()).select_from(DbSession))).scalar() or 0
    n_objects = (await db.execute(select(func.count()).select_from(Object))).scalar() or 0
    n_human = (await db.execute(select(func.count()).where(Object.source == "human"))).scalar() or 0
    err = {k: int(n) for k, n in (await db.execute(
        select(ErrorCandidate.kind, func.count()).where(ErrorCandidate.status == "pending").group_by(ErrorCandidate.kind))).all()}
    scen = {k: int(n) for k, n in (await db.execute(
        select(ScenarioCandidate.kind, func.count()).where(ScenarioCandidate.state == "pending").group_by(ScenarioCandidate.kind))).all()}
    return {
        "size": {"sessions": int(n_sessions), "objects": int(n_objects), "human_labeled": int(n_human)},
        "class_balance": {"missing": len(coverage["class_balance"]["missing"]), "rare": len(coverage["class_balance"]["rare"])},
        "coverage_gaps": coverage["gaps"], "scene_coverage": coverage["scene_coverage"],
        "fix_queue": err, "fix_queue_total": sum(err.values()),
        "scenarios": scen, "geo": coverage["geo"],
    }
