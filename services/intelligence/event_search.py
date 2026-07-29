"""Ask the corpus a question about behaviour, rather than one session at a time.

Every route the event layer shipped with is scoped to a session, which is the right shape for reviewing a
drive and the wrong shape for the question the events exist to answer. "Show me every illegal lane change
where the signal was red" is not a fact about a session; it is a fact about the fleet, and until now there
was no way to ask it.

Two things make that more than a filtered list.

The first is co-occurrence. The interesting behaviours are conjunctions: a crossing *while* a signal was
red, a weave *near* a junction. Those are a temporal join within a session, and doing it in SQL rather than
by pulling both sets into Python is the difference between a query and a report generator.

The second is that a result has to be reachable. An event with a track and a frame is a place to go; an
event that is only a row is a statistic. Every hit carries the session, the frame and the actor, so the
answer is a list of things to look at.
"""

from __future__ import annotations

import uuid as _uuid

from core.logging import get_logger

log = get_logger("event_search")

# Bounded because a question that matches half the corpus is a question that needs narrowing, and returning
# 200,000 rows to prove it helps nobody.
MAX_LIMIT = 1000


def _as_uuid(v):
    if v in (None, ""):
        return None
    return v if isinstance(v, _uuid.UUID) else _uuid.UUID(str(v))


async def search_events(
    db,
    *,
    kinds: list[str] | None = None,
    severities: list[str] | None = None,
    states: list[str] | None = None,
    city: str | None = None,
    session_id=None,
    min_conf: float | None = None,
    # Co-occurrence: keep a hit only when an event of `with_kind` overlaps it in time in the same session.
    with_kind: str | None = None,
    # And, for signals, only when that overlapping phase was in one of these states. "a crossing while the
    # signal was red" is the whole point, and without this it would only be "a crossing near any signal".
    with_payload_state: list[str] | None = None,
    within_ns: int = 0,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """Behaviour across the corpus, narrowed, with the context needed to go and look at it."""
    from sqlalchemy import String, and_, cast, func, or_, select

    from db.models import Frame, TimelineEvent, Track
    from db.models import Session as DbSession
    from services.intelligence.event_taxonomy import kinds as taxonomy_kinds
    from services.intelligence.event_taxonomy import severity_of

    limit = max(1, min(int(limit), MAX_LIMIT))

    ev = TimelineEvent
    q = (select(ev, DbSession.city, DbSession.vehicle_id, Track.class_id)
         .join(DbSession, DbSession.session_id == ev.session_id)
         .outerjoin(Track, Track.track_id == ev.track_id)
         .where(ev.modality == "driving"))

    if kinds:
        q = q.where(ev.kind.in_(kinds))
    if severities:
        # Severity lives in the taxonomy rather than on the row, so a reclassification does not need a data
        # migration. Expanded to the kinds that carry it, which keeps the filter in SQL where the limit is.
        wanted = [k for k in taxonomy_kinds() if severity_of(k) in severities]
        if not wanted:
            return _empty(severities)
        q = q.where(ev.kind.in_(wanted))
    if states:
        q = q.where(ev.state.in_(states))
    if city:
        q = q.where(DbSession.city == city)
    if session_id:
        q = q.where(ev.session_id == _as_uuid(session_id))
    if min_conf is not None:
        q = q.where(or_(ev.conf.is_(None), ev.conf >= float(min_conf)))

    if with_kind:
        # The co-occurrence join. Two events overlap when neither ends before the other starts, with a point
        # event treated as the instant [t, t]; `within_ns` widens both sides so "while" can mean "around".
        other = ev.__table__.alias("other")
        o_start = other.c.t_start_ns - within_ns
        o_end = func.coalesce(other.c.t_end_ns, other.c.t_start_ns) + within_ns
        e_end = func.coalesce(ev.t_end_ns, ev.t_start_ns)
        cond = and_(
            other.c.session_id == ev.session_id,
            other.c.event_id != ev.event_id,
            other.c.kind == with_kind,
            ev.t_start_ns <= o_end,
            o_start <= e_end,
        )
        if with_payload_state:
            cond = and_(cond, cast(other.c.payload["state"].astext, String).in_(with_payload_state))
        q = q.where(select(other.c.event_id).where(cond).exists())

    total = (await db.execute(
        select(func.count()).select_from(q.order_by(None).subquery()))).scalar() or 0

    rows = (await db.execute(
        q.order_by(ev.conf.desc().nullslast(), ev.t_start_ns).limit(limit).offset(offset))).all()

    from services.autolabel.ontology import get_ontology
    onto = get_ontology()

    # Frame timestamps let a hit be shown as an offset into its drive rather than as epoch nanoseconds,
    # which is the difference between a scrub target and a number.
    starts: dict = {}
    if rows:
        sids = {r[0].session_id for r in rows}
        for sid, t0 in (await db.execute(
                select(Frame.session_id, func.min(Frame.ts_ns))
                .where(Frame.session_id.in_(sids)).group_by(Frame.session_id))).all():
            starts[sid] = int(t0 or 0)

    out = []
    for e, city_name, vehicle, class_id in rows:
        origin = starts.get(e.session_id)
        out.append({
            "event_id": str(e.event_id), "kind": e.kind, "severity": severity_of(e.kind),
            "session_id": str(e.session_id), "city": city_name, "vehicle_id": vehicle,
            "frame_id": str(e.frame_id) if e.frame_id else None,
            "track_id": str(e.track_id) if e.track_id else None,
            "actor_class": (onto.by_id(class_id).name if class_id else None),
            "t_start_ns": e.t_start_ns, "t_end_ns": e.t_end_ns,
            "at_s": (round((e.t_start_ns - origin) / 1e9, 2) if origin is not None else None),
            "duration_s": (round(((e.t_end_ns or e.t_start_ns) - e.t_start_ns) / 1e9, 2)),
            "conf": e.conf, "state": e.state, "payload": e.payload,
        })

    log.info("event_search", kinds=kinds, with_kind=with_kind, matched=total, returned=len(out))
    return {"total": total, "returned": len(out), "offset": offset, "limit": limit,
            "results": out}


def _empty(severities) -> dict:
    return {"total": 0, "returned": 0, "offset": 0, "limit": 0, "results": [],
            "detail": f"no event kind carries severity {severities}"}


async def corpus_summary(db) -> dict:
    """What the corpus holds, so a question can be aimed rather than guessed at."""
    from sqlalchemy import func, select

    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from services.intelligence.event_taxonomy import severity_of

    rows = (await db.execute(
        select(TimelineEvent.kind, TimelineEvent.state, func.count())
        .where(TimelineEvent.modality == "driving")
        .group_by(TimelineEvent.kind, TimelineEvent.state))).all()
    cities = (await db.execute(
        select(DbSession.city, func.count(func.distinct(TimelineEvent.session_id)))
        .join(TimelineEvent, TimelineEvent.session_id == DbSession.session_id)
        .where(TimelineEvent.modality == "driving")
        .group_by(DbSession.city))).all()

    by_kind: dict = {}
    by_state: dict = {}
    by_sev: dict = {}
    for kind, state, n in rows:
        by_kind[kind] = by_kind.get(kind, 0) + int(n)
        by_state[state] = by_state.get(state, 0) + int(n)
        sev = severity_of(kind)
        by_sev[sev] = by_sev.get(sev, 0) + int(n)

    return {"total": sum(by_kind.values()),
            "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
            "by_state": dict(sorted(by_state.items())),
            "by_severity": dict(sorted(by_sev.items())),
            "cities": {str(c): int(n) for c, n in cities if c},
            "sessions_with_events": len({r for r in (await db.execute(
                select(func.distinct(TimelineEvent.session_id))
                .where(TimelineEvent.modality == "driving"))).scalars()})}
