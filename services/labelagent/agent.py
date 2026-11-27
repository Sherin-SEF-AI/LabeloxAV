"""The label agent: checks each machine label's class with independent specialists, and fixes, flags, asks
or keeps.

The ladder per object, all from vectors already stored, so no model runs per object:

1. **Corpus memory** - what people called the nearest things they have ruled on.
2. **Learned probe** - a linear head on SigLIP 2 + DINOv3 trained on those rulings.
3. **Family zero-shot** - SigLIP 2 text over the object's confusion-family shortlist, with negatives.

and the decision, with thresholds measured on the human-ruling benchmark (core/config.py LabelAgentSettings):

- **fix**: memory and probe agree on a different class, each with its margin, and the zero-shot does not
  prefer something else. The class changes, the object stays in review, and the change is reversible.
- **flag**: the same agreement, on "not an object". The box is never deleted; it goes to review with the
  question "is this a real object?".
- **ask**: memory and probe agree at the weaker margins. The class does not change; the question ("sign or
  hoarding?") and the suggestion ride along to the reviewer.
- **keep**: anything else.

Three things it never does, whatever the votes: accept a label, touch an object a person made or ruled on,
or change an object without recording how to put it back. Any non-keep decision on an auto-accepted label
moves it back to review, which withholds acceptance and is all it may do to acceptance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.labelagent.families import ConfusionGraph, shortlist
from services.labelagent.memory import Bank, Probe, Vectors, Vote, load_vectors
from services.labelagent.truth import (
    MACHINE_REVIEWERS,
    NOT_OBJECT,
    NOT_OBJECT_CLASS_NAMES,
    load_all,
)
from services.labelagent.zeroshot import ZsAnswer, classify

log = get_logger("labelagent.agent")

VERSION = "labelagent-v1"
KIND = "labelagent"

KEEP, FIX, FLAG, ASK = "keep", "fix", "flag", "ask"


@dataclass
class Votes:
    memory: Vote | None = None
    probe: tuple[int | None, float, float] | None = None     # label, probability, margin
    zs: ZsAnswer | None = None

    def as_trace(self, name_of) -> dict:
        out: dict = {}
        if self.memory is not None and self.memory.label is not None:
            out["memory"] = {"label": name_of(self.memory.label), "margin": self.memory.margin,
                             "share": self.memory.share, "neighbours": len(self.memory.neighbours)}
        if self.probe is not None and self.probe[0] is not None:
            out["probe"] = {"label": name_of(self.probe[0]), "p": self.probe[1], "margin": self.probe[2]}
        if self.zs is not None and self.zs.label is not None:
            out["zeroshot"] = {"label": name_of(self.zs.label), "margin": self.zs.margin}
        return out


@dataclass
class Decision:
    action: str
    label: int | None = None
    question: str = ""
    reasons: list[str] = field(default_factory=list)


def decide(pred: int, votes: Votes, policy, name_of) -> Decision:
    """The action for one object, from its votes. Pure, so every threshold is testable without a model."""
    m, p, z = votes.memory, votes.probe, votes.zs
    if m is None or m.label is None or p is None or p[0] is None:
        return Decision(KEEP, reasons=["a specialist had no opinion"])
    if m.label != p[0]:
        return Decision(KEEP, reasons=[f"memory says {name_of(m.label)}, probe says {name_of(p[0])}"])
    agreed = m.label
    if agreed == pred:
        return Decision(KEEP, reasons=[f"memory and probe both confirm {name_of(pred)}"])

    def question() -> str:
        if agreed == NOT_OBJECT:
            return f"Is this a real object? The specialists think this {name_of(pred)} box is not one."
        return f"{name_of(pred)} or {name_of(agreed)}?"

    reasons = [f"memory {name_of(agreed)} (margin {m.margin:.2f})", f"probe {name_of(agreed)} (margin {p[2]:.2f})"]
    zs_against = z is not None and z.label is not None and z.label != agreed and z.margin >= policy.zs_veto_margin
    if z is not None and z.label is not None:
        reasons.append(f"zero-shot {name_of(z.label)} (margin {z.margin:.2f})")

    if m.margin >= policy.fix_memory_margin and p[2] >= policy.fix_probe_margin and not zs_against:
        return Decision(FLAG if agreed == NOT_OBJECT else FIX, agreed, question(), reasons)
    if m.margin >= policy.ask_memory_margin and p[2] >= policy.ask_probe_margin:
        if zs_against:
            reasons.append("zero-shot disagrees, so asking instead of fixing")
        return Decision(ASK, agreed, question(), reasons)
    return Decision(KEEP, reasons=reasons + ["agreement too weak to act on"])


# ------------------------------------------------------------------------------------------------ specialists

@dataclass
class Specialists:
    bank: Bank
    probe: Probe | None
    graph: ConfusionGraph
    n_rulings: int
    stamp: tuple


_cache: dict[str, Specialists] = {}


async def _stamp(db: AsyncSession) -> tuple:
    """Changes whenever a person rules on anything, which is when memory and the probe must refresh: a
    correction saved a second ago is already a neighbour."""
    from db.models import Review

    row = (await db.execute(select(func.count(Review.review_id), func.max(Review.ts_ns))
                            .where(Review.reviewer.notin_(MACHINE_REVIEWERS)))).first()
    return (int(row[0] or 0), int(row[1] or 0))


async def load_specialists(db: AsyncSession, policy) -> Specialists:
    stamp = await _stamp(db)
    cached = _cache.get("s")
    if cached is not None and cached.stamp == stamp:
        return cached
    items = await load_all(db)
    vectors = await load_vectors(db, [i.object_id for i in items])
    probe = None
    if len(items) >= policy.min_rulings:
        try:
            probe = Probe.fit(items, vectors)
        except ValueError as exc:
            log.warning("labelagent.probe_unfit", error=str(exc))
    spec = Specialists(bank=Bank.build(items, vectors), probe=probe, graph=ConfusionGraph.build(items),
                       n_rulings=len(items), stamp=stamp)
    _cache["s"] = spec
    log.info("labelagent.specialists_loaded", rulings=len(items), probe=probe is not None)
    return spec


def votes_for(object_id: str, pred: int, vectors: Vectors, spec: Specialists, onto, *, k: int = 15) -> Votes:
    q = vectors.siglip.get(object_id)
    if q is None:
        return Votes()
    v = Votes(memory=spec.bank.vote(q, k=k, exclude_object=object_id))
    if spec.probe is not None:
        v.probe = spec.probe.predict(object_id, vectors)
    if pred != NOT_OBJECT:
        v.zs = classify(q, shortlist(pred, spec.graph, onto), onto)
    return v


# ------------------------------------------------------------------------------------------------ session pass

def _name_of(onto):
    def name_of(cid: int | None) -> str:
        if cid is None:
            return "?"
        if cid == NOT_OBJECT:
            return "not an object"
        try:
            return onto.by_id(cid).name
        except KeyError:
            return f"class {cid}"
    return name_of


async def candidates(db: AsyncSession, session_id: uuid.UUID, limit: int | None = None) -> list:
    """The session's machine labels the agent may look at: not made by a person, not settled (accepted or
    rejected), and never ruled on by a person, whatever state that left them in."""
    from db.models import Frame, Object, Review

    ruled = exists().where(Review.object_id == Object.object_id, Review.reviewer.notin_(MACHINE_REVIEWERS))
    stmt = (select(Object).join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id, Object.source != "human",
                   Object.state.in_(("review", "auto_accept", "annotate")), ~ruled)
            .order_by(Frame.ts_ns, Object.object_id))
    if limit:
        stmt = stmt.limit(limit)
    return list((await db.execute(stmt)).scalars())


async def run_session(db: AsyncSession, session_id: str | uuid.UUID, *, apply: bool = False,
                      limit: int | None = None, created_by: str | None = None, policy=None) -> dict:
    """Check every candidate label in a session. With `apply`, write the decisions as one reversible
    AgentRun and commit; without it, change nothing and report what would happen."""
    from core.config import get_settings
    from db.models import AgentRun
    from services.autolabel.ontology import get_ontology

    policy = policy or get_settings().labelagent
    sid = uuid.UUID(str(session_id))
    onto = get_ontology()
    name_of = _name_of(onto)
    not_ids = {c.id for c in onto.classes if c.name in NOT_OBJECT_CLASS_NAMES}

    objs = await candidates(db, sid, limit)
    if not objs:
        return {"session_id": str(sid), "objects": 0, "decisions": {}, "applied": False,
                "detail": "no machine labels in this session are open for checking"}
    spec = await load_specialists(db, policy)
    vectors = await load_vectors(db, [str(o.object_id) for o in objs])
    from services.labelagent.report import fixes_allowed

    may_fix, gate_reason = fixes_allowed(policy)

    counts: dict[str, int] = {}
    decided: list[tuple] = []
    no_vector = 0
    for o in objs:
        oid = str(o.object_id)
        if oid not in vectors.siglip:
            no_vector += 1
            continue
        pred = NOT_OBJECT if o.class_id in not_ids else int(o.class_id)
        votes = votes_for(oid, pred, vectors, spec, onto, k=policy.memory_k)
        d = decide(pred, votes, policy, name_of)
        if d.action == FIX and not may_fix:
            d = Decision(ASK, d.label, d.question, d.reasons + [gate_reason])
        counts[d.action] = counts.get(d.action, 0) + 1
        if d.action != KEEP:
            decided.append((o, d, votes))

    samples = [{"object_id": str(o.object_id), "action": d.action, "from": name_of(o.class_id),
                "to": name_of(d.label), "question": d.question, "reasons": d.reasons}
               for o, d, _v in decided[:25]]
    out = {"session_id": str(sid), "objects": len(objs), "no_vector": no_vector, "decisions": counts,
           "fixes_allowed": may_fix, "gate": gate_reason,
           "rulings_in_memory": spec.n_rulings, "probe": spec.probe is not None, "samples": samples,
           "applied": False}
    if not apply or not decided:
        return out

    run = AgentRun(kind=KIND, scope={"session_id": str(sid)}, status="committed",
                   policy={"version": VERSION, **policy.model_dump()}, counts=counts, created_by=created_by)
    db.add(run)
    await db.flush()
    changes: dict[str, dict] = {}
    for o, d, votes in decided:
        ch: dict = {"action": d.action}
        if d.action == FIX and d.label is not None:
            ch["from_class"] = int(o.class_id)
            o.class_id = d.label
        if o.state == "auto_accept":
            ch["from_state"] = "auto_accept"
            o.state = "review"
        if "from_class" in ch or "from_state" in ch:
            o.version = (o.version or 0) + 1
        prov = dict(o.provenance or {})
        prov["agent_run_id"] = str(run.run_id)
        prov["labelagent"] = {"version": VERSION, "action": d.action, "from": name_of(ch.get("from_class", o.class_id)),
                              "to": name_of(d.label), "question": d.question, "reasons": d.reasons,
                              "votes": votes.as_trace(name_of)}
        o.provenance = prov
        changes[str(o.object_id)] = ch
    run.changes = changes
    await db.commit()
    log.info("labelagent.applied", session_id=str(sid), run_id=str(run.run_id), **counts)
    return {**out, "applied": True, "run_id": str(run.run_id)}


async def undo_object(db: AsyncSession, object_id: str | uuid.UUID) -> dict:
    """Undo the agent's change to one object: the class and state it had before, and its trace removed.

    Refused once a person has taken the object over (its source is then `human`), or when a later agent run
    owns it, for the same reason a whole-run revert skips those: undoing would overwrite somebody's decision.
    """
    from db.models import AgentRun, Object

    obj = await db.get(Object, uuid.UUID(str(object_id)))
    if obj is None:
        raise LookupError("object not found")
    prov = dict(obj.provenance or {})
    run_id = prov.get("agent_run_id")
    if "labelagent" not in prov or not run_id:
        raise ValueError("the label agent has not changed this object")
    if obj.source == "human":
        raise ValueError("a person has edited this object since; their label stands")
    run = await db.get(AgentRun, uuid.UUID(str(run_id)))
    if run is None or run.kind != KIND or run.status != "committed":
        raise ValueError("the agent run that changed this object can no longer be undone")
    changes = dict(run.changes or {})
    ch = dict(changes.get(str(obj.object_id)) or {})
    if "from_class" in ch:
        obj.class_id = ch["from_class"]
    if "from_state" in ch:
        obj.state = ch["from_state"]
    obj.version = (obj.version or 0) + 1
    prov.pop("labelagent", None)
    prov.pop("agent_run_id", None)
    obj.provenance = prov
    changes[str(obj.object_id)] = {**ch, "undone": True}
    run.changes = changes
    await db.commit()
    return {"object_id": str(obj.object_id), "class_id": obj.class_id, "state": obj.state}
