"""Derive driving events for a session and persist them without ever duplicating one.

Idempotence is the whole design problem here. Deriving is cheap and will be re-run: after a lane is
corrected, after signal states are relabelled, after a threshold in the taxonomy moves. If a re-run appended,
the second run would double every event and the third would triple it, and the rates every mining and safety
surface computes would drift upward with nothing but the number of times somebody pressed the button.

So a derived event has an identity that is not its row id: (kind, track, start). Re-deriving updates the
payload and confidence of a matching row and inserts only what is genuinely new. Events a human has ruled on
are left exactly as they are, for the same reason the reasoner's rerun leaves them alone: a person's decision
is the output of the loop, and a machine that revisits it is undoing the work the loop exists to collect.

Stale candidates are the other half. If a re-derivation no longer produces an event that a previous run
proposed, the geometry changed and the old candidate is wrong. Leaving it would mean a corrected lane still
shows the lane change it no longer implies, so unreviewed candidates that no longer derive are removed. Only
unreviewed ones, and the count is reported rather than done quietly.
"""

from __future__ import annotations

import uuid as _uuid

from core.logging import get_logger
from services.intelligence.event_taxonomy import TaxonomyError, kind_spec, validate

log = get_logger("driving_events")

MODALITY = "driving"


def _identity(kind: str, track_id, t_start_ns: int) -> tuple:
    """What makes two derivations the same event.

    Start time rather than the interval, because a re-derivation that sees one more frame of a lane change
    legitimately extends the end while being the same crossing. Keying on the interval would make every such
    extension a new event.
    """
    return (str(kind), str(track_id) if track_id else None, int(t_start_ns))


async def derive_events(db, session_id) -> list[dict]:
    """Every derived candidate for a session, from every deriver, un-persisted."""
    from services.intelligence.lane_events import detect_lane_events
    from services.intelligence.signal_events import detect_signal_events

    out: list[dict] = []
    for name, fn in (("lane", detect_lane_events), ("signal", detect_signal_events)):
        try:
            out.extend(await fn(db, session_id))
        except Exception as exc:  # noqa: BLE001
            # One deriver failing must not cost the session the other's events. The failure is loud and the
            # partial result is honest about being partial.
            log.exception("driving_events.deriver_failed", deriver=name,
                          session=str(session_id), error=str(exc))
    return out


async def persist_driving_events(db, session_id, *, prune_stale: bool = True) -> dict:
    """Derive and reconcile against what is already stored.

    Returns counts rather than rows: a session can produce thousands of signal phases, and the caller that
    wants them has a list endpoint that paginates.
    """
    from sqlalchemy import select

    from db.models import TimelineEvent

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))
    candidates = await derive_events(db, sid)

    existing = (await db.execute(
        select(TimelineEvent).where(TimelineEvent.session_id == sid,
                                    TimelineEvent.modality == MODALITY,
                                    TimelineEvent.source == "auto"))).scalars().all()
    by_identity = {_identity(e.kind, e.track_id, e.t_start_ns): e for e in existing}

    inserted = updated = unchanged = skipped_reviewed = rejected = 0
    seen: set[tuple] = set()

    for c in candidates:
        try:
            validate(c["kind"], t_start_ns=c["t_start_ns"], t_end_ns=c.get("t_end_ns"),
                     track_id=c.get("track_id"), frame_id=c.get("frame_id"), source="auto")
        except TaxonomyError as exc:
            # A deriver producing something the taxonomy forbids is a bug in the deriver, and the right
            # response is to drop the event loudly rather than write a record no consumer can interpret.
            log.error("driving_events.rejected", kind=c.get("kind"), reason=str(exc))
            rejected += 1
            continue

        ident = _identity(c["kind"], c.get("track_id"), c["t_start_ns"])
        seen.add(ident)
        row = by_identity.get(ident)
        if row is None:
            db.add(TimelineEvent(
                session_id=sid, kind=c["kind"], modality=MODALITY,
                t_start_ns=c["t_start_ns"], t_end_ns=c.get("t_end_ns"),
                track_id=_as_uuid(c.get("track_id")), frame_id=_as_uuid(c.get("frame_id")),
                conf=c.get("conf"), payload=c.get("payload") or {},
                source="auto", state="review",
                provenance={"deriver": c["kind"].split("_")[0],
                            "severity": (kind_spec(c["kind"]) or {}).get("severity", "info")}))
            inserted += 1
            continue

        if row.state in ("confirmed", "rejected"):
            skipped_reviewed += 1
            continue

        changed = (row.t_end_ns != c.get("t_end_ns") or row.payload != (c.get("payload") or {})
                   or row.conf != c.get("conf"))
        if not changed:
            unchanged += 1
            continue
        row.t_end_ns = c.get("t_end_ns")
        row.payload = c.get("payload") or {}
        row.conf = c.get("conf")
        row.frame_id = _as_uuid(c.get("frame_id"))
        row.version += 1
        updated += 1

    pruned = 0
    if prune_stale:
        for ident, row in by_identity.items():
            if ident in seen or row.state in ("confirmed", "rejected"):
                continue
            await db.delete(row)
            pruned += 1

    await db.commit()

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1

    log.info("driving_events.persisted", session=str(sid), derived=len(candidates),
             inserted=inserted, updated=updated, pruned=pruned)
    return {"session_id": str(sid), "derived": len(candidates), "inserted": inserted,
            "updated": updated, "unchanged": unchanged, "pruned_stale": pruned,
            "skipped_reviewed": skipped_reviewed, "rejected_by_taxonomy": rejected,
            "by_kind": dict(sorted(counts.items()))}


def _as_uuid(value):
    if value is None:
        return None
    return value if isinstance(value, _uuid.UUID) else _uuid.UUID(str(value))


async def session_event_summary(db, session_id) -> dict:
    """What a session's driving events add up to: counts by kind, severity, and review state.

    The shape the session header and the mining filters read, so neither has to pull every event to show a
    badge.
    """
    from sqlalchemy import func, select

    from db.models import TimelineEvent
    from services.intelligence.event_taxonomy import severity_of

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))
    rows = (await db.execute(
        select(TimelineEvent.kind, TimelineEvent.state, func.count())
        .where(TimelineEvent.session_id == sid, TimelineEvent.modality == MODALITY)
        .group_by(TimelineEvent.kind, TimelineEvent.state))).all()

    by_kind: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for kind, state, n in rows:
        by_kind[kind] = by_kind.get(kind, 0) + int(n)
        by_state[state] = by_state.get(state, 0) + int(n)
        sev = severity_of(kind)
        by_severity[sev] = by_severity.get(sev, 0) + int(n)

    return {"session_id": str(sid), "total": sum(by_kind.values()),
            "by_kind": dict(sorted(by_kind.items())),
            "by_state": dict(sorted(by_state.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "violations": by_severity.get("violation", 0)}
