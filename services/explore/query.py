"""The explorer's predicate, evaluated in SQL.

`services/curation/slices.py:matches_predicate` already defines the clause vocabulary for a saved cohort, but
it is a pure Python matcher over one pre-assembled record: `materialize_slice` has to stream every frame and
object into memory to use it. That is fine for verifying membership and far too expensive to drive an
interactive explorer over a 32k-frame corpus.

So the same vocabulary gets a second evaluator here that pushes every clause into SQL. One predicate shape,
two evaluators: the pure one stays the testable definition of membership, this one makes it fast. Clause names
are kept identical so a `CurationSlice.predicate` saved by the explorer still works in the existing export and
materialize paths, and vice versa.

Vocabulary (every clause optional, all clauses AND together, a missing clause is unconstrained):
    weather / time_of_day / road_type / density : list[str]  -> frame.scene axes
    cities        : list[str]    -> session.city
    class_names   : list[str]    -> object.class_id via the ontology
    states        : list[str]    -> object.state
    sources       : list[str]    -> object.source
    min_conf      : float        -> object.conf >=
    max_conf      : float        -> object.conf <=
    tags          : list[str]    -> object.tags, any-of
    frame_tags    : list[str]    -> frame.tags, any-of
    session_id    : str
    object_ids    : list[str]    -> explicit selection (lasso)
    frame_ids     : list[str]    -> explicit selection (lasso)
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Select, or_, select
from sqlalchemy.sql.elements import ColumnElement

from db.models import Frame, Object
from db.models import Session as DbSession

_SCENE_AXES = ("weather", "time_of_day", "road_type", "density")


def _any_tag(col, tags: list[str]) -> ColumnElement:
    """any-of over a JSONB array column. `contains` maps to the @> operator, which uses the GIN index added in
    migration 0063; OR-ing single-element containment gives any-of semantics while staying indexable."""
    return or_(*[col.contains([t]) for t in tags])


def _class_ids(class_names: list[str]) -> list[int]:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ids = []
    for n in class_names:
        try:
            ids.append(onto.by_name(n).id)
        except Exception:  # noqa: BLE001 - an unknown class name simply matches nothing
            continue
    return ids


def scene_clauses(pred: dict) -> list[ColumnElement]:
    """frame.scene JSONB axis clauses. Backed by the ix_frame_scene_gin index (migration 0062)."""
    out = []
    for axis in _SCENE_AXES:
        want = pred.get(axis)
        if want:
            out.append(Frame.scene[axis].astext.in_(list(want)))
    return out


def object_clauses(pred: dict) -> list[ColumnElement]:
    """Clauses that constrain the Object row itself."""
    out: list[ColumnElement] = []
    if pred.get("class_names"):
        out.append(Object.class_id.in_(_class_ids(list(pred["class_names"]))))
    if pred.get("states"):
        out.append(Object.state.in_(list(pred["states"])))
    if pred.get("sources"):
        out.append(Object.source.in_(list(pred["sources"])))
    if pred.get("min_conf") is not None:
        out.append(Object.conf >= float(pred["min_conf"]))
    if pred.get("max_conf") is not None:
        out.append(Object.conf <= float(pred["max_conf"]))
    if pred.get("tags"):
        out.append(_any_tag(Object.tags, list(pred["tags"])))
    if pred.get("object_ids"):
        out.append(Object.object_id.in_([UUID(str(v)) for v in pred["object_ids"]]))
    return out


def frame_clauses(pred: dict) -> list[ColumnElement]:
    """Clauses on the frame and its session."""
    out: list[ColumnElement] = list(scene_clauses(pred))
    if pred.get("frame_tags"):
        out.append(_any_tag(Frame.tags, list(pred["frame_tags"])))
    if pred.get("session_id"):
        out.append(Frame.session_id == UUID(str(pred["session_id"])))
    if pred.get("frame_ids"):
        out.append(Frame.frame_id.in_([UUID(str(v)) for v in pred["frame_ids"]]))
    if pred.get("cities"):
        out.append(DbSession.city.in_(list(pred["cities"])))
    return out


def _needs_session_join(pred: dict) -> bool:
    return bool(pred.get("cities"))


def object_select(pred: dict, *columns) -> Select:
    """A SELECT over objects (joined to frame, and to session only when a clause needs it) with the whole
    predicate applied. Pass the columns you want; defaults to the Object entity."""
    # select_from(Object) is explicit so the join still has a left side when `columns` is an aggregate such
    # as func.count(), which carries no entity for SQLAlchemy to infer one from.
    stmt = (select(*(columns or (Object,))).select_from(Object)
            .join(Frame, Frame.frame_id == Object.frame_id))
    if _needs_session_join(pred):
        stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id)
    for c in object_clauses(pred) + frame_clauses(pred):
        stmt = stmt.where(c)
    return stmt


def frame_select(pred: dict, *columns) -> Select:
    """A SELECT over frames with the whole predicate applied. Object-level clauses become an EXISTS over the
    frame's objects, so a frame matches when ANY of its objects does (the same any-of semantics
    `matches_predicate` uses for classes and states)."""
    stmt = select(*(columns or (Frame,))).select_from(Frame)
    if _needs_session_join(pred):
        stmt = stmt.join(DbSession, DbSession.session_id == Frame.session_id)
    for c in frame_clauses(pred):
        stmt = stmt.where(c)
    obj_cs = object_clauses(pred)
    if obj_cs:
        sub = select(Object.object_id).where(Object.frame_id == Frame.frame_id)
        for c in obj_cs:
            sub = sub.where(c)
        stmt = stmt.where(sub.exists())
    return stmt
