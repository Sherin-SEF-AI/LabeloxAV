"""The lineage graph: sessions to labels to gold to trainset to model to promotion, as one DAG.

Every edge in this graph is already recorded. A dataset commit knows its slice spec and its parent, a
training job knows its dataset and its gold set, a model registry row knows the job it came from, a
promotion knows the model and the evaluation that cleared it. What has never existed is the graph: each
fact is readable from its own table and the chain from a shipped model back to the sessions its training
data came from could only be walked by hand, one query at a time.

That chain is the answer to the questions that actually get asked in an audit. Which footage is this model
made of. Was any of it from a subject who has since withdrawn consent. Which promotion introduced the
regression this slice is showing. None of those are hard given the graph and all of them are hours of
manual work without it.

Two deliberate limits:

- **Nodes are resolved lazily and bounded.** A full corpus graph is tens of thousands of sessions and
  nobody reads that; the graph is built outward from one node to a depth, which is how it is used.
- **A missing edge is a node marked incomplete, not a silent omission.** A trainset whose gold set has
  been deleted should render as a broken link, because that is a real fact about the lineage rather than
  an absence to be smoothed over.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("lineage_graph")

# The kinds of thing in the graph, in the order work flows through them. Used to lay the graph out, so a
# renderer does not have to know the domain.
NODE_KINDS = ("session", "dataset", "gold", "training_job", "model", "promotion", "deployment")


@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: dict = field(default_factory=dict)
    # True when an edge points at something that is no longer there. Rendered as a break rather than
    # dropped, because a missing gold set is a fact about the lineage.
    incomplete: bool = False


@dataclass
class Edge:
    source: str
    target: str
    kind: str


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> str:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        return node.id

    def add_edge(self, source: str, target: str, kind: str) -> None:
        if source and target and not any(
                e.source == source and e.target == target and e.kind == kind for e in self.edges):
            self.edges.append(Edge(source, target, kind))

    def as_dict(self) -> dict:
        order = {k: i for i, k in enumerate(NODE_KINDS)}
        return {
            "nodes": [{"id": n.id, "kind": n.kind, "label": n.label, "meta": n.meta,
                       "incomplete": n.incomplete,
                       # The column a renderer puts it in. Kept here so the layout is the same in every
                       # client rather than reinvented per view.
                       "rank": order.get(n.kind, len(NODE_KINDS))}
                      for n in self.nodes.values()],
            "edges": [{"source": e.source, "target": e.target, "kind": e.kind} for e in self.edges],
            "kinds": list(NODE_KINDS),
        }


def _nid(kind: str, ident: str) -> str:
    return f"{kind}:{ident}"


async def model_lineage(db: AsyncSession, model_version: str, *, max_sessions: int = 40) -> dict:
    """Everything a model is made of, walked backwards from the model itself.

    The direction that matters for an audit: given something that shipped, what is in it. The forward
    direction (given this footage, what shipped) is `session_lineage` below.
    """
    from db.models import DatasetCommit, GoldSet, ModelRegistry, TrainingJob

    g = Graph()
    model = (await db.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == model_version)
    )).scalars().first()
    if model is None:
        raise ValueError(f"model {model_version!r} is not in the registry")

    model_node = g.add_node(Node(_nid("model", model_version), "model", model_version,
                                 {"task": model.task, "is_champion": bool(model.is_champion),
                                  "metrics": dict(model.gold_metrics or {}),
                                  "weights_uri": model.weights_uri}))

    if model.is_champion:
        promo = g.add_node(Node(_nid("promotion", model_version), "promotion",
                                f"promoted {model_version}",
                                {"promoted_from": model.promoted_from}))
        g.add_edge(model_node, promo, "promoted_to")
    if model.promoted_from:
        # The model it beat. A promotion is a comparison, and a lineage that shows only the winner cannot
        # answer "what did this replace".
        g.add_node(Node(_nid("model", model.promoted_from), "model", model.promoted_from,
                        {"superseded": True}))
        g.add_edge(_nid("model", model.promoted_from), model_node, "superseded_by")

    # The registry records its dataset commit directly, so this edge is exact rather than inferred.
    commit = await db.get(DatasetCommit, model.dataset_commit) if model.dataset_commit else None
    commit_node = None
    if model.dataset_commit:
        commit_node = g.add_node(Node(
            _nid("dataset", model.dataset_commit), "dataset",
            (f"{(commit.slice_spec or {}).get('name') or 'dataset'} {model.dataset_commit[:8]}"
             if commit else f"{model.dataset_commit[:8]} (gone)"),
            {"objects": getattr(commit, "object_count", None),
             "ontology": getattr(commit, "ontology_version", None)},
            incomplete=commit is None))
        g.add_edge(commit_node, model_node, "trained_on")
        if commit is not None and commit.parent_id:
            g.add_node(Node(_nid("dataset", commit.parent_id), "dataset",
                            f"{commit.parent_id[:8]}", {}))
            g.add_edge(_nid("dataset", commit.parent_id), commit_node, "parent_of")

    # The gold set the metrics were measured on, when the run recorded one.
    gold_id = (model.gold_metrics or {}).get("gold_id")
    if gold_id:
        gold = await db.get(GoldSet, gold_id)
        gold_node = g.add_node(Node(
            _nid("gold", gold_id), "gold", gold.name if gold else f"{str(gold_id)[:12]} (gone)",
            {"n_objects": getattr(gold, "n_objects", None),
             "tracks_sealed": bool(getattr(gold, "tracks_sealed", False))},
            incomplete=gold is None))
        g.add_edge(gold_node, model_node, "evaluated_on")

    job = (await db.execute(
        select(TrainingJob).where(TrainingJob.purpose.isnot(None))
        .order_by(TrainingJob.created_at.desc()).limit(50))).scalars().all()
    matched = next((j for j in job
                    if str((j.result or {}).get("model_version") or "") == model_version), None)
    if matched is not None:
        job_node = g.add_node(Node(_nid("training_job", str(matched.job_id)), "training_job",
                                   matched.purpose or str(matched.job_id)[:8],
                                   {"task_type": matched.task_type, "status": matched.status,
                                    "counts": dict(matched.counts or {})}))
        g.add_edge(job_node, model_node, "produced")
        if commit_node:
            g.add_edge(commit_node, job_node, "trained_on")

    spec = dict((matched.dataset_spec or {}) if matched is not None else {})
    sessions = await _sessions_for_spec(db, spec, limit=max_sessions)
    for sid, label, meta in sessions:
        session_node = g.add_node(Node(_nid("session", sid), "session", label, meta))
        g.add_edge(session_node, commit_node or model_node, "contributed_to")

    return {**g.as_dict(), "root": model_node,
            # Said out loud: a graph truncated at forty sessions must not read as a model built from forty.
            "sessions_shown": len(sessions), "sessions_truncated": len(sessions) >= max_sessions}


async def session_lineage(db: AsyncSession, session_id: str) -> dict:
    """The forward direction: given this footage, what did it end up in.

    The question an erasure request asks. If a subject withdraws consent for a session, this is what has to
    be re-examined, and answering it by hand meant reading every dataset commit's slice spec.
    """
    from db.models import DatasetCommit, Frame, Object
    from db.models import Session as DbSession

    g = Graph()
    sess = await db.get(DbSession, uuid.UUID(session_id))
    if sess is None:
        raise ValueError(f"session {session_id} not found")

    from sqlalchemy import func

    n_frames = (await db.execute(
        select(func.count()).select_from(Frame)
        .where(Frame.session_id == sess.session_id))).scalar_one()
    n_objects = (await db.execute(
        select(func.count()).select_from(Object)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == sess.session_id))).scalar_one()

    root = g.add_node(Node(_nid("session", session_id), "session",
                           f"{sess.vehicle_id or sess.city or 'session'} {session_id[:8]}",
                           {"frames": int(n_frames), "objects": int(n_objects),
                            "city": sess.city, "pack_id": sess.pack_id}))

    commits = (await db.execute(
        select(DatasetCommit).order_by(DatasetCommit.created_at.desc()).limit(200))).scalars().all()
    reached = 0
    for commit in commits:
        spec = dict(commit.slice_spec or {})
        # A commit includes this session if its spec names it, or names its city, or names nothing (which
        # means the whole corpus). The last case is the one a manual check misses.
        names_session = str(session_id) in str(spec.get("session_id") or "")
        names_city = bool(sess.city and sess.city in (spec.get("cities") or []))
        whole_corpus = not spec.get("session_id") and not spec.get("cities")
        if not (names_session or names_city or whole_corpus):
            continue
        node = g.add_node(Node(_nid("dataset", commit.commit_id), "dataset",
                               f"{spec.get('name') or 'dataset'} {commit.commit_id[:8]}",
                               {"objects": commit.object_count,
                                "match": ("named" if names_session else
                                          "by city" if names_city else "whole corpus")}))
        g.add_edge(root, node, "contributed_to")
        reached += 1
        if reached >= 20:
            break

    return {**g.as_dict(), "root": root, "datasets_reached": reached,
            "detail": ("a commit whose slice names no session and no city covers the whole corpus, so it "
                       "includes this session even though it never mentions it")}


async def _sessions_for_spec(db: AsyncSession, spec: dict, *, limit: int) -> list[tuple[str, str, dict]]:
    """Which sessions a dataset spec's data came from, ranked by how much each contributed."""
    from sqlalchemy import func

    from db.models import Frame, Object
    from db.models import Session as DbSession

    stmt = (select(DbSession.session_id, DbSession.vehicle_id, DbSession.city,
                   func.count(Object.object_id))
            .select_from(DbSession)
            .join(Frame, Frame.session_id == DbSession.session_id)
            .join(Object, Object.frame_id == Frame.frame_id)
            .where(Object.state.in_(spec.get("states") or ["accepted", "approved"]))
            .group_by(DbSession.session_id, DbSession.vehicle_id, DbSession.city)
            .order_by(func.count(Object.object_id).desc())
            .limit(limit))
    if spec.get("cities"):
        stmt = stmt.where(DbSession.city.in_(spec["cities"]))
    rows = (await db.execute(stmt)).all()
    return [(str(sid), f"{veh or city or 'session'} {str(sid)[:8]}",
             {"objects": int(n), "city": city, "vehicle_id": veh})
            for sid, veh, city, n in rows]


async def dataset_lineage(db: AsyncSession, commit_id: str) -> dict:
    """One dataset commit, its ancestors, and the models trained from its line."""
    from db.models import DatasetCommit, ModelRegistry

    g = Graph()
    commit = await db.get(DatasetCommit, commit_id)
    if commit is None:
        raise ValueError(f"dataset commit {commit_id} not found")

    root = g.add_node(Node(_nid("dataset", commit_id), "dataset",
                           f"{(commit.slice_spec or {}).get('name') or 'dataset'} {commit_id[:8]}",
                           {"objects": commit.object_count,
                            "ontology": commit.ontology_version}))

    cur, depth = commit, 0
    while cur is not None and cur.parent_id and depth < 20:
        parent = await db.get(DatasetCommit, cur.parent_id)
        node = g.add_node(Node(
            _nid("dataset", cur.parent_id), "dataset",
            f"{cur.parent_id[:8]}" if parent else f"{cur.parent_id[:8]} (gone)",
            {"objects": getattr(parent, "object_count", None)}, incomplete=parent is None))
        g.add_edge(node, _nid("dataset", cur.commit_id), "parent_of")
        cur, depth = parent, depth + 1

    # Exact, not inferred: the registry stores the commit each model was trained on.
    models = (await db.execute(
        select(ModelRegistry).where(ModelRegistry.dataset_commit == commit_id))).scalars().all()
    for m in models:
        node = g.add_node(Node(_nid("model", m.model_version), "model", m.model_version,
                               {"is_champion": bool(m.is_champion), "task": m.task}))
        g.add_edge(root, node, "trained")

    return {**g.as_dict(), "root": root, "ancestry_depth": depth,
            "models": len(models)}
