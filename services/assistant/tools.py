"""What the assistant can do. Reading tools run immediately; changing tools only propose.

Every reading tool answers within what the caller may see (services/project_access.py). Every changing tool
resolves its effect now, for the preview, and stores it as an AssistantProposal with exactly what it would
touch; nothing changes until a reviewer approves, and approval runs the change through the path a person
would use (the reversible, audited NL edit; the auto-label start route with its GPU and health checks; the
error-detection run). What was approved is what changes: a reclassify carries the object ids it previewed,
not the filter that found them.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    AssistantProposal,
    AutolabelJob,
    DatasetCommit,
    ErrorCandidate,
    ExportJob,
    Frame,
    ImportJob,
    LabelProject,
    ModelRegistry,
    Object,
    TrainingJob,
)
from db.models import Session as DbSession
from services.autolabel.ontology import get_ontology
from services.project_access import scope_visible_sessions

MAX_PREVIEW = 300  # objects one reclassify or flag proposal may carry


@dataclass
class Context:
    db: AsyncSession
    user: Any
    hidden: frozenset
    request: str = ""
    session_id: str | None = None   # where the person is working, if anywhere
    frame_id: str | None = None
    proposals: list[dict] = field(default_factory=list)


class ToolError(ValueError):
    """A bad argument the model can correct: the message goes back to it as the tool result."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    kind: str  # "read" | "propose"
    handler: Callable[[Context, dict], Awaitable[dict]]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


_STR = {"type": "string"}
_STRS = {"type": "array", "items": {"type": "string"}}


def _class_ids(names: list[str] | None) -> list[int]:
    """Class names to ids, accepting aliases and loose spelling; an unknown name is a ToolError with hints."""
    onto = get_ontology()
    ids = []
    for raw in names or []:
        n = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if onto.has_name(n):
            ids.append(onto.by_name(n).id)
            continue
        hit = next((c for c in onto.classes if n in (a.lower() for a in c.aliases)), None)
        if hit is not None:
            ids.append(hit.id)
            continue
        near = [c.name for c in onto.classes if n[:4] and n[:4] in c.name][:6]
        raise ToolError(f"no class named {raw!r}" + (f"; did you mean one of {near}?" if near else
                                                     "; call list_classes to find the right name"))
    return ids


async def _visible_session(ctx: Context, raw: str | None) -> uuid.UUID | None:
    if not raw:
        return None
    try:
        sid = uuid.UUID(str(raw))
    except ValueError:
        raise ToolError(f"{raw!r} is not a session id; call find_sessions to look one up") from None
    sess = await ctx.db.get(DbSession, sid)
    if sess is None or (sess.project_id is not None and sess.project_id in ctx.hidden):
        raise ToolError(f"session {raw} not found")
    return sid


async def _visible_frame(ctx: Context, raw: str | None) -> uuid.UUID | None:
    if not raw:
        return None
    try:
        fid = uuid.UUID(str(raw))
    except ValueError:
        raise ToolError(f"{raw!r} is not a frame id") from None
    frame = await ctx.db.get(Frame, fid)
    if frame is None:
        raise ToolError(f"frame {raw} not found")
    await _visible_session(ctx, str(frame.session_id))
    return fid


# ---- reading ----------------------------------------------------------------------------------------------

async def list_classes(ctx: Context, a: dict) -> dict:
    onto = get_ontology()
    q = (a.get("query") or "").strip().lower().replace(" ", "_")
    hits = [c for c in onto.classes
            if not q or q in c.name or q in c.l1 or any(q in al.lower() for al in c.aliases)]
    return {"classes": [{"name": c.name, "group": c.l1} for c in hits[:40]], "total_matching": len(hits)}


async def count_objects(ctx: Context, a: dict) -> dict:
    onto = get_ontology()
    q = (select(Object.class_id, Object.state, func.count())
         .join(Frame, Frame.frame_id == Object.frame_id).join(DbSession, DbSession.session_id == Frame.session_id))
    q = scope_visible_sessions(q, ctx.hidden)
    if ids := _class_ids(a.get("class_names")):
        q = q.where(Object.class_id.in_(ids))
    if a.get("states"):
        q = q.where(Object.state.in_(a["states"]))
    if sid := await _visible_session(ctx, a.get("session_id")):
        q = q.where(Frame.session_id == sid)
    if a.get("max_confidence") is not None:
        q = q.where(Object.conf <= float(a["max_confidence"]))
    rows = (await ctx.db.execute(q.group_by(Object.class_id, Object.state))).all()
    by_class: dict[str, dict[str, int]] = {}
    for cid, state, n in rows:
        by_class.setdefault(onto.by_id(cid).name, {})[state] = int(n)
    total = sum(sum(v.values()) for v in by_class.values())
    top = sorted(by_class.items(), key=lambda kv: -sum(kv[1].values()))[:25]
    return {"total": total, "by_class": dict(top), "classes_shown": len(top), "classes_total": len(by_class)}


# Sessions record a city as the airport code a fleet uses (DEL, BLR), and people ask by name.
CITY_CODES = {
    "delhi": "DEL", "new delhi": "DEL", "bengaluru": "BLR", "bangalore": "BLR", "mumbai": "BOM", "bombay": "BOM",
    "chennai": "MAA", "madras": "MAA", "hyderabad": "HYD", "kolkata": "CCU", "calcutta": "CCU", "pune": "PNQ",
    "ahmedabad": "AMD", "jaipur": "JAI", "lucknow": "LKO", "kochi": "COK", "cochin": "COK", "cuttack": "CTC",
    "bhubaneswar": "BBI", "chandigarh": "IXC", "goa": "GOI", "coimbatore": "CJB", "indore": "IDR",
}


async def find_sessions(ctx: Context, a: dict) -> dict:
    q = scope_visible_sessions(select(DbSession), ctx.hidden)
    if text := (a.get("query") or "").strip():
        like = f"%{text}%"
        match: list[ColumnElement[bool]] = [DbSession.vehicle_id.ilike(like), DbSession.city.ilike(like),
                                            DbSession.route.ilike(like)]
        if code := CITY_CODES.get(text.lower()):
            match += [DbSession.city == code, DbSession.vehicle_id.ilike(f"%-{code}-%")]
        q = q.where(or_(*match))
    rows = (await ctx.db.execute(q.order_by(DbSession.start_ts_ns.desc()).limit(int(a.get("limit") or 10)))).scalars().all()
    frames: dict[uuid.UUID, int] = {}
    if rows:
        frames = {s: int(n) for s, n in (await ctx.db.execute(
            select(Frame.session_id, func.count()).where(Frame.session_id.in_([s.session_id for s in rows]))
            .group_by(Frame.session_id))).all()}
    out: dict = {"sessions": [{"session_id": str(s.session_id), "vehicle": s.vehicle_id, "city": s.city,
                               "route": s.route, "frames": int(frames.get(s.session_id, 0))} for s in rows]}
    if not rows and text:
        # Cities are often stored as codes (DEL, BLR), so a city name misses. Say what exists, so the
        # model can retry with the right spelling instead of telling the person there is nothing.
        cities = (await ctx.db.execute(scope_visible_sessions(
            select(DbSession.city).where(DbSession.city.isnot(None)).distinct(), ctx.hidden).limit(60))).scalars().all()
        out["no_match"] = f"nothing matched {text!r}; cities on record: {sorted(cities)}"
    return out


async def session_overview(ctx: Context, a: dict) -> dict:
    sid = await _visible_session(ctx, a.get("session_id") or ctx.session_id)
    if sid is None:
        raise ToolError("which session? give session_id, or call find_sessions")
    onto = get_ontology()
    sess = await ctx.db.get(DbSession, sid)
    frames = (await ctx.db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))).scalar_one()
    states: dict[str, int] = {st: int(n) for st, n in (await ctx.db.execute(
        select(Object.state, func.count()).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == sid).group_by(Object.state))).all()}
    top = (await ctx.db.execute(
        select(Object.class_id, func.count()).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == sid).group_by(Object.class_id).order_by(func.count().desc()).limit(8))).all()
    project = await ctx.db.get(LabelProject, sess.project_id) if sess and sess.project_id else None
    return {"session_id": str(sid), "vehicle": sess.vehicle_id if sess else None, "city": sess.city if sess else None,
            "project": project.name if project else None, "frames": int(frames),
            "objects_by_state": {k: int(v) for k, v in states.items()},
            "top_classes": {onto.by_id(c).name: int(n) for c, n in top}}


async def queue_overview(ctx: Context, a: dict) -> dict:
    onto = get_ontology()
    q = (select(Object.class_id, func.count()).join(Frame, Frame.frame_id == Object.frame_id)
         .join(DbSession, DbSession.session_id == Frame.session_id).where(Object.state.in_(("review", "annotate"))))
    q = scope_visible_sessions(q, ctx.hidden)
    if sid := await _visible_session(ctx, a.get("session_id")):
        q = q.where(Frame.session_id == sid)
    if ids := _class_ids(a.get("class_names")):
        q = q.where(Object.class_id.in_(ids))
    rows = (await ctx.db.execute(q.group_by(Object.class_id).order_by(func.count().desc()))).all()
    return {"waiting_for_review": sum(int(n) for _, n in rows),
            "by_class": {onto.by_id(c).name: int(n) for c, n in rows[:15]}}


async def error_overview(ctx: Context, a: dict) -> dict:
    rows = (await ctx.db.execute(select(ErrorCandidate.kind, func.count())
                                 .where(ErrorCandidate.status == "pending").group_by(ErrorCandidate.kind))).all()
    return {"pending_by_detector": {k: int(n) for k, n in rows},
            "note": "Track consistency and confident learning are the detectors measured to catch wrong classes."}


async def list_datasets(ctx: Context, a: dict) -> dict:
    rows = (await ctx.db.execute(select(DatasetCommit).order_by(DatasetCommit.created_at.desc())
                                 .limit(int(a.get("limit") or 10)))).scalars().all()
    return {"datasets": [{"name": (d.slice_spec or {}).get("name"), "commit_id": d.commit_id,
                          "objects": d.object_count, "formats": (d.slice_spec or {}).get("formats", []),
                          "created": d.created_at.date().isoformat() if d.created_at else None} for d in rows]}


async def model_status(ctx: Context, a: dict) -> dict:
    champs = (await ctx.db.execute(select(ModelRegistry).where(ModelRegistry.is_champion.is_(True)))).scalars().all()
    jobs = (await ctx.db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(5))).scalars().all()
    return {
        "champions": [{"model": m.model_version, "task": m.task,
                       "gold_map50": (m.gold_metrics or {}).get("map50"), "promoted_from": m.promoted_from}
                      for m in champs],
        "recent_training": [{"job_id": str(j.job_id), "status": j.status, "stage": j.stage,
                             "purpose": j.purpose, "map50": (j.metrics or {}).get("map50"),
                             "promoted": (j.result or {}).get("promoted"), "error": j.error,
                             "started": j.created_at.date().isoformat() if j.created_at else None} for j in jobs],
    }


async def list_projects(ctx: Context, a: dict) -> dict:
    from services.project_classes import describe

    counts = {pid: int(n) for pid, n in (await ctx.db.execute(
        select(DbSession.project_id, func.count()).group_by(DbSession.project_id))).all()}
    rows = (await ctx.db.execute(select(LabelProject).order_by(LabelProject.name))).scalars().all()
    return {"projects": [{"project_id": str(p.project_id), "name": p.name, "sessions": counts.get(p.project_id, 0),
                          "visible_to": "members only" if p.visibility == "members" else "whole team",
                          "classes": describe(p.class_ids)["class_set"]}
                         for p in rows if p.project_id not in ctx.hidden]}


async def job_overview(ctx: Context, a: dict) -> dict:
    out: dict = {}
    for kind, model in (("autolabel", AutolabelJob), ("import", ImportJob), ("training", TrainingJob),
                        ("export", ExportJob)):
        by = {st: int(n) for st, n in (await ctx.db.execute(
            select(model.status, func.count()).group_by(model.status))).all()}
        failed = (await ctx.db.execute(select(model.error).where(model.status == "error")
                                       .order_by(model.created_at.desc()).limit(3))).scalars().all()
        out[kind] = {"by_status": by, "recent_errors": [e for e in failed if e]}
    return out


# ---- proposing --------------------------------------------------------------------------------------------

async def _store(ctx: Context, tool: str, args: dict, summary: str, preview: dict) -> dict:
    p = AssistantProposal(tool=tool, args=args, summary=summary, preview=preview, request=ctx.request or None,
                          created_by=getattr(ctx.user, "user_id", None))
    ctx.db.add(p)
    await ctx.db.commit()
    out = {"proposal_id": str(p.proposal_id), "tool": tool, "summary": summary, "preview": preview,
           "status": "pending"}
    ctx.proposals.append(out)
    return {"proposal_id": out["proposal_id"], "status": "waiting for a reviewer to approve",
            "summary": summary, "would_change": preview.get("count")}


async def _resolve_edit(ctx: Context, a: dict, operation: str) -> tuple[dict, list[dict], dict[str, int]]:
    from services.agent.nl_edit import resolve

    onto = get_ontology()
    select_ids = _class_ids(a.get("from_classes") or a.get("class_names"))
    if not select_ids:
        raise ToolError("name the classes to change (from_classes)")
    to_id = None
    if operation == "reclassify":
        to = _class_ids([a.get("to_class", "")]) if a.get("to_class") else []
        if not to:
            raise ToolError("reclassify needs to_class")
        to_id = to[0]
    fid = await _visible_frame(ctx, a.get("frame_id"))
    sid = None if fid else await _visible_session(ctx, a.get("session_id") or ctx.session_id)
    if fid is None and sid is None:
        raise ToolError("a change needs a scope: give session_id or frame_id (find_sessions finds one)")
    plan = {"operation": operation, "select_class_ids": select_ids, "to_class_id": to_id, "mask_missing": False,
            "attr": None, "referential": False,
            "raw": f"{operation} {', '.join(onto.by_id(i).name for i in select_ids)}"
                   + (f" to {onto.by_id(to_id).name}" if to_id else "")}
    res = await resolve(ctx.db, plan, frame_id=fid, session_id=sid, limit=MAX_PREVIEW)
    states = set(a.get("states") or [])
    objs = [o for o in res["objects"] if not states or o["state"] in states]
    # Human labels are never changed by a language edit (nl_edit refuses them); say so in the preview.
    human = {str(i) for i in (await ctx.db.execute(select(Object.object_id).where(
        Object.object_id.in_([uuid.UUID(o["object_id"]) for o in objs]), Object.source == "human"))).scalars()}
    kept = [o for o in objs if o["object_id"] not in human]
    by_class: dict[str, int] = {}
    for o in kept:
        by_class[o["class_name"]] = by_class.get(o["class_name"], 0) + 1
    scope = {"frame_id": str(fid)} if fid else {"session_id": str(sid)}
    return {**plan, **scope}, kept, {"human_labels_left_alone": len(human), **{f"from_{k}": v for k, v in by_class.items()}}


async def propose_reclassify(ctx: Context, a: dict) -> dict:
    plan, objs, counts = await _resolve_edit(ctx, a, "reclassify")
    if not objs:
        return {"proposal_id": None, "status": "nothing matched; no proposal made", **counts}
    onto = get_ontology()
    summary = f"Change {len(objs)} label(s) to {onto.by_id(plan['to_class_id']).name} " \
              f"({', '.join(f'{v} {k[5:]}' for k, v in counts.items() if k.startswith('from_'))}), sent back to review"
    return await _store(ctx, "propose_reclassify", {"plan": plan, "object_ids": [o["object_id"] for o in objs]},
                        summary, {"count": len(objs), **counts, "sample": objs[:12]})


async def propose_flag_for_review(ctx: Context, a: dict) -> dict:
    plan, objs, counts = await _resolve_edit(ctx, a, "flag")
    if not objs:
        return {"proposal_id": None, "status": "nothing matched; no proposal made", **counts}
    summary = f"Send {len(objs)} label(s) back to review " \
              f"({', '.join(f'{v} {k[5:]}' for k, v in counts.items() if k.startswith('from_'))})"
    return await _store(ctx, "propose_flag_for_review", {"plan": plan, "object_ids": [o["object_id"] for o in objs]},
                        summary, {"count": len(objs), **counts, "sample": objs[:12]})


async def propose_autolabel(ctx: Context, a: dict) -> dict:
    sid = await _visible_session(ctx, a.get("session_id") or ctx.session_id)
    if sid is None:
        raise ToolError("which session? give session_id")
    sess = await ctx.db.get(DbSession, sid)
    frames = (await ctx.db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))).scalar_one()
    only_new = bool(a.get("only_unlabelled", True))
    summary = f"Auto-label session {sess.vehicle_id if sess else sid} ({frames} frames" \
              + (", unlabelled frames only)" if only_new else ")")
    return await _store(ctx, "propose_autolabel", {"session_id": str(sid), "only_unlabelled": only_new},
                        summary, {"count": int(frames), "unit": "frames"})


async def propose_error_detection(ctx: Context, a: dict) -> dict:
    sid = await _visible_session(ctx, a.get("session_id") or ctx.session_id)
    summary = "Look for likely label errors " + (f"in session {sid}" if sid else "across the corpus")
    return await _store(ctx, "propose_error_detection", {"session_id": str(sid) if sid else None},
                        summary, {"count": None})


async def _visible_project(ctx: Context, a: dict) -> LabelProject | None:
    raw = a.get("project_id") or a.get("project_name")
    if not raw:
        return None
    try:
        p = await ctx.db.get(LabelProject, uuid.UUID(str(raw)))
    except ValueError:
        p = (await ctx.db.execute(select(LabelProject).where(LabelProject.name.ilike(str(raw))))).scalars().first()
    if p is None or p.project_id in ctx.hidden:
        raise ToolError(f"no project {raw!r}; call list_projects")
    return p


async def propose_export(ctx: Context, a: dict) -> dict:
    from services.export.dataset import SUPPORTED_EXPORT_FORMATS

    formats = [f.lower() for f in (a.get("formats") or ["coco"])]
    bad = [f for f in formats if f not in SUPPORTED_EXPORT_FORMATS]
    if bad:
        raise ToolError(f"unknown format(s) {bad}; one of {sorted(SUPPORTED_EXPORT_FORMATS)}")
    sid = await _visible_session(ctx, a.get("session_id") or ctx.session_id)
    q = (select(func.count()).select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
         .join(DbSession, DbSession.session_id == Frame.session_id).where(Object.state != "rejected"))
    q = scope_visible_sessions(q, ctx.hidden)
    if sid:
        q = q.where(Frame.session_id == sid)
    n = int((await ctx.db.execute(q)).scalar_one())
    name = (a.get("name") or (f"session-{str(sid)[:8]}" if sid else "dataset")).strip()[:120]
    summary = f"Export {n:,} labels as {', '.join(formats)} to a dataset named {name!r}" \
              + (f" from session {sid}" if sid else " from every session you can see")
    return await _store(ctx, "propose_export", {"name": name, "formats": formats,
                                                "session_id": str(sid) if sid else None},
                        summary, {"count": n, "unit": "labels"})


async def propose_training(ctx: Context, a: dict) -> dict:
    project = await _visible_project(ctx, a)
    classes = [c.name for c in (get_ontology().by_id(i) for i in _class_ids(a.get("class_names")))]
    epochs = max(1, min(int(a.get("epochs") or 20), 200))
    q = (select(func.count()).select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
         .join(DbSession, DbSession.session_id == Frame.session_id).where(Object.state != "rejected"))
    q = scope_visible_sessions(q, ctx.hidden)
    if project is not None:
        q = q.where(DbSession.project_id == project.project_id)
    n = int((await ctx.db.execute(q)).scalar_one())
    spec: dict = {}
    if project is not None:
        spec["project_id"] = str(project.project_id)
    if classes:
        spec["include_classes"] = classes
    summary = f"Train a detector for {epochs} epochs on " + (f"project {project.name}" if project else "every project") \
              + (f", classes {', '.join(classes)}" if classes else "") + f" (up to {n:,} labels). It is not promoted automatically."
    return await _store(ctx, "propose_training", {"dataset_spec": spec, "epochs": epochs}, summary,
                        {"count": n, "unit": "labels"})


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("list_classes", "Find ontology class names by a word, a group (e.g. two_wheeler, vru) or an alias. Use "
         "it before naming a class you are not sure of.", _schema({"query": _STR}), "read", list_classes),
    Tool("count_objects", "Count labels, by class and review state, optionally for some classes, states "
         "(review, annotate, accepted, auto_accept, rejected), one session, or at most a confidence.",
         _schema({"class_names": _STRS, "states": _STRS, "session_id": _STR, "max_confidence": {"type": "number"}}),
         "read", count_objects),
    Tool("find_sessions", "Find recorded drives (sessions) by vehicle, city or route text, newest first.",
         _schema({"query": _STR, "limit": {"type": "integer"}}), "read", find_sessions),
    Tool("session_overview", "Frames, labels by state and the most common classes of one session.",
         _schema({"session_id": _STR}), "read", session_overview),
    Tool("queue_overview", "How many labels wait for review, by class, overall or for one session, optionally "
         "for some classes only.", _schema({"session_id": _STR, "class_names": _STRS}), "read", queue_overview),
    Tool("error_overview", "Pending likely-label-error candidates, by detector.", _schema({}), "read", error_overview),
    Tool("list_datasets", "Recently exported datasets: name, how many labels, formats.",
         _schema({"limit": {"type": "integer"}}), "read", list_datasets),
    Tool("model_status", "The current champion detector and the most recent training runs with their results.",
         _schema({}), "read", model_status),
    Tool("list_projects", "Projects, how many sessions each holds, who can see it and which classes it labels.",
         _schema({}), "read", list_projects),
    Tool("job_overview", "Import, auto-label, training and export jobs by status, with recent errors.",
         _schema({}), "read", job_overview),
    Tool("propose_export", "Propose exporting a dataset (formats such as coco, yolo, parquet), of one session or of "
         "everything. Creates a proposal; exports nothing.",
         _schema({"name": _STR, "formats": _STRS, "session_id": _STR}), "propose", propose_export),
    Tool("propose_training", "Propose training a detector, optionally on one project (project_id or project_name) "
         "and some classes. Creates a proposal; starts nothing. Never promotes the result.",
         _schema({"project_id": _STR, "project_name": _STR, "class_names": _STRS, "epochs": {"type": "integer"}}),
         "propose", propose_training),
    Tool("propose_reclassify", "Propose changing machine labels of some classes to another class, in one session "
         "or frame. Creates a proposal for a reviewer; changes nothing itself. Human labels are never changed.",
         _schema({"from_classes": _STRS, "to_class": _STR, "session_id": _STR, "frame_id": _STR, "states": _STRS},
                 ["from_classes", "to_class"]), "propose", propose_reclassify),
    Tool("propose_flag_for_review", "Propose sending machine labels of some classes back to human review, in one "
         "session or frame. Creates a proposal; changes nothing itself.",
         _schema({"class_names": _STRS, "session_id": _STR, "frame_id": _STR, "states": _STRS}, ["class_names"]),
         "propose", propose_flag_for_review),
    Tool("propose_autolabel", "Propose running auto-labelling on one session. Creates a proposal; starts nothing.",
         _schema({"session_id": _STR, "only_unlabelled": {"type": "boolean"}}, ["session_id"]), "propose",
         propose_autolabel),
    Tool("propose_error_detection", "Propose running the label-error detectors, on one session or everywhere. "
         "Creates a proposal; runs nothing.", _schema({"session_id": _STR}), "propose", propose_error_detection),
]}


# ---- approving --------------------------------------------------------------------------------------------

async def execute(db: AsyncSession, p: AssistantProposal, user: Any) -> dict:
    """Run an approved proposal through the path a person would use. Raises on failure."""
    created_by = str(user.user_id) if getattr(user, "user_id", None) else None
    if p.tool in ("propose_reclassify", "propose_flag_for_review"):
        from services.agent.nl_edit import apply_edit

        res = await apply_edit(db, p.args["plan"], [uuid.UUID(i) for i in p.args["object_ids"]],
                               created_by=created_by)
        if "error" in res:
            raise ToolError(res["error"])
        return res
    if p.tool == "propose_autolabel":
        from services.api.deps import AutolabelStartIn
        from services.api.routers.autolabel import start

        return await start(AutolabelStartIn(session_id=p.args["session_id"],
                                            only_unlabelled=bool(p.args.get("only_unlabelled", True))), db)
    if p.tool == "propose_error_detection":
        from services.errordetect.queue import run_detection

        return await run_detection(db, p.args.get("session_id"))
    if p.tool == "propose_export":
        from services.api.deps import ExportIn
        from services.api.routers.datasets import start_export

        return await start_export(ExportIn(name=p.args["name"], formats=p.args["formats"],
                                           session_id=p.args.get("session_id")), db)
    if p.tool == "propose_training":
        from services.api.deps import TrainingStartIn
        from services.api.routers.training import start as start_training

        return await start_training(TrainingStartIn(purpose="perception", task_type="detection",
                                                    dataset_spec=p.args["dataset_spec"],
                                                    hparams={"epochs": p.args["epochs"]}, promote=False,
                                                    notes="proposed by the assistant"),
                                    user=user, hidden=frozenset())
    raise ToolError(f"unknown proposal tool {p.tool}")


__all__ = ["MAX_PREVIEW", "TOOLS", "Context", "Tool", "ToolError", "execute"]
