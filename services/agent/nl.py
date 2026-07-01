"""Natural-language control. A person types an instruction ("auto-accept the two-wheelers above 0.9",
"double-check all riders", "how many pedestrians here", "undo that") and it becomes a scoped agent action.

The parser is deterministic and rule-based: it resolves class groups and synonyms against the ontology, a
confidence threshold, and an action verb, then dispatches to the agent primitives already built (plan,
commit, reconcile, revert). It is intentionally not a black box -- the resolved intent is returned
alongside the result so the person sees exactly what the agent understood before, and after, it acts.
Structured so an LLM parse can fill the same Intent for fuzzier phrasing without changing the executor.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun, Object
from services.agent.frame_agent import commit_frame, plan_frame
from services.agent.policy import PolicyThresholds

# Phrase -> ontology l1 categories it expands to. Resolving against the ontology's own taxonomy (rather than
# a hardcoded name list) keeps a group correct as the ontology grows: "two-wheelers" is whatever the
# ontology files under l1 == two_wheeler, today and tomorrow.
_VEHICLE_L1 = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy"}
_L1_GROUPS: dict[str, set[str]] = {
    "two-wheeler": {"two_wheeler"}, "two wheeler": {"two_wheeler"}, "2-wheeler": {"two_wheeler"},
    "three-wheeler": {"three_wheeler"}, "three wheeler": {"three_wheeler"}, "autorickshaw": {"three_wheeler"},
    "rickshaw": {"three_wheeler"}, "four-wheeler": {"four_wheeler"},
    "heavy vehicle": {"heavy"}, "trucks": {"heavy"},
    "animals": {"animal"}, "animal": {"animal"},
    "vulnerable": {"vru"}, "vru": {"vru"}, "people": {"vru"},
    "vehicles": _VEHICLE_L1, "vehicle": _VEHICLE_L1, "all vehicles": _VEHICLE_L1,
}


@dataclass
class Intent:
    action: str = "plan"                 # plan | accept | reconcile | find | revert
    class_names: set[str] = field(default_factory=set)
    class_ids: set[int] = field(default_factory=set)
    conf_min: float | None = None
    all_classes: bool = True             # no class filter given -> every machine object

    def to_dict(self) -> dict:
        return {"action": self.action, "classes": sorted(self.class_names) or "all",
                "conf_min": self.conf_min}


def _resolve_conf(text: str) -> float | None:
    m = re.search(r"(?:above|over|>=?|greater than|at least)\s*(\d+(?:\.\d+)?)\s*(%?)", text)
    if not m:
        m = re.search(r"\b(0?\.\d+)\b", text)  # a bare 0.9
        if not m:
            return None
        return float(m.group(1))
    val = float(m.group(1))
    if m.group(2) == "%" or val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def parse_command(text: str, onto) -> Intent:
    t = text.lower().strip()
    names_in_onto = {c.name for c in onto.classes}

    # action verb
    if re.search(r"\b(undo|revert|roll ?back)\b", t):
        action = "revert"
    elif re.search(r"\b(auto[- ]?accept|accept|approve|commit|confirm)\b", t):
        action = "accept"
    elif re.search(r"\b(reconcile|double[- ]?check|second opinion|verify class|re-?classify)\b", t):
        action = "reconcile"
    elif re.search(r"\b(how many|count|find|list|show|which)\b", t):
        action = "find"
    else:
        action = "plan"

    # classes: l1-category group phrases first, then any exact ontology class name mentioned
    class_names: set[str] = set()
    l1_wanted: set[str] = set()
    for phrase, cats in _L1_GROUPS.items():
        if phrase in t:
            l1_wanted |= cats
    if l1_wanted:
        class_names |= {c.name for c in onto.classes if c.l1 in l1_wanted}
    for name in names_in_onto:
        spaced = re.escape(name.replace("_", " "))
        # tolerate a trailing plural 's' so "pedestrians"/"riders" match the class "pedestrian"/"rider"
        if re.search(rf"\b{spaced}s?\b", t) or re.search(rf"\b{re.escape(name)}s?\b", t):
            class_names.add(name)

    class_ids = {c.id for c in onto.classes if c.name in class_names}
    return Intent(action=action, class_names=class_names, class_ids=class_ids,
                  conf_min=_resolve_conf(t), all_classes=not class_ids)


async def _latest_committed_run(db: AsyncSession, frame_id: uuid.UUID) -> AgentRun | None:
    rows = await db.execute(
        select(AgentRun).where(AgentRun.kind == "frame", AgentRun.status == "committed")
        .order_by(AgentRun.created_at.desc())
    )
    for r in rows.scalars().all():
        if (r.scope or {}).get("frame_id") == str(frame_id):
            return r
    return None


async def execute_command(db: AsyncSession, text: str, frame_id: uuid.UUID, created_by: str | None = None,
                          can_write: bool = True) -> dict:
    """Parse and run a natural-language instruction against one frame. Returns the intent, result, and a
    human-readable summary. plan/find are read-only; accept/revert write (and are reversible). Write actions
    require can_write (reviewer role); otherwise they are refused, not silently downgraded."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    intent = parse_command(text, onto)
    only = intent.class_ids or None
    if intent.action in ("accept", "revert") and not can_write:
        return {"intent": intent.to_dict(), "result": None, "blocked": True,
                "summary": f"'{intent.action}' changes labels and needs reviewer role"}
    th = PolicyThresholds(auto_accept_conf=intent.conf_min) if intent.conf_min is not None else PolicyThresholds()
    scope_txt = (", ".join(sorted(intent.class_names)) if intent.class_names else "all objects")

    if intent.action == "find":
        q = select(Object).where(Object.frame_id == frame_id, Object.source != "human")
        if only:
            q = q.where(Object.class_id.in_(list(only)))
        objs = list((await db.execute(q)).scalars().all())
        if intent.conf_min is not None:
            objs = [o for o in objs if float(o.conf) >= intent.conf_min]
        by_state: dict[str, int] = {}
        for o in objs:
            by_state[o.state] = by_state.get(o.state, 0) + 1
        return {"intent": intent.to_dict(), "result": {"count": len(objs), "by_state": by_state},
                "summary": f"{len(objs)} {scope_txt} on this frame ({', '.join(f'{k}:{v}' for k, v in by_state.items()) or 'none'})"}

    if intent.action == "plan":
        res = await plan_frame(db, frame_id, th, only)
        c = res["counts"]
        return {"intent": intent.to_dict(), "result": res["counts"],
                "summary": f"would auto-accept {c['auto_accept']}, review {c['review']}, annotate {c['annotate']} of {scope_txt} (dry-run)"}

    if intent.action == "accept":
        res = await commit_frame(db, frame_id, th, created_by=created_by, only_classes=only)
        c = res["counts"]
        return {"intent": intent.to_dict(), "result": res,
                "summary": f"applied {res['applied']} changes to {scope_txt}: auto-accepted {c['auto_accept']}, routed {c['review']} to review (run {res['run_id'][:8]}, reversible)"}

    if intent.action == "reconcile":
        from services.agent.reconcile import reconcile_frame

        q = select(Object.object_id).where(Object.frame_id == frame_id, Object.source != "human")
        if only:
            q = q.where(Object.class_id.in_(list(only)))
        ids = [str(x) for x in (await db.execute(q)).scalars().all()]
        res = await reconcile_frame(db, frame_id, ids or None)
        v = res.get("verdicts", {})
        return {"intent": intent.to_dict(), "result": res,
                "summary": f"reconciled {res['reconciled']} {scope_txt}: {v.get('confirm', 0)} confirmed, {v.get('correct', 0)} to relabel, {v.get('unsure', 0)} unsure"}

    if intent.action == "revert":
        from services.agent.runs import revert_run

        run = await _latest_committed_run(db, frame_id)
        if run is None:
            return {"intent": intent.to_dict(), "result": None, "summary": "nothing to revert on this frame"}
        res = await revert_run(db, run.run_id)
        return {"intent": intent.to_dict(), "result": res,
                "summary": f"reverted the last agent run on this frame ({res['reverted']} objects restored)"}

    return {"intent": intent.to_dict(), "result": None, "summary": "did not understand that command"}
