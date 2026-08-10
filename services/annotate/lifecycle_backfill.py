"""Backfill `Object.lifecycle`, and be explicit about how little of it is derived.

Review is append-only and its `before` snapshot deliberately keeps the pre-human `source` and `conf`, so for
an object a person ruled on the history is genuinely recoverable. For everything else it is not: the agent
path writes only AgentRun.changes, the 3D edit path keeps no Review row at all, and `Object.version` is a
counter with no snapshot behind it. On this corpus that means well under one percent is derived and the rest
is defaulted.

That ratio is the whole reason this reports rather than just writes. A migration that quietly stamps a
lifecycle onto half a million rows produces a column that looks authoritative and mostly is not, and the
next person to read it has no way to tell which is which. So every row records how its value was reached, and
the dry run prints the split before anything is written.

  derived    a Review row says what a human did and when
  inferred   no review, but state and source agree on a machine story
  defaulted  nothing recoverable; machine_proposed with the reason recorded
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("lifecycle_backfill")

MACHINE_SOURCES = frozenset({
    "fused", "auto_accept", "imported", "recall", "interpolated", "interp", "propagated", "relabel",
})
ENDORSING = frozenset({"confirm", "accept"})
EDITING = frozenset({"reclassify", "reclassify_track", "adjust_geometry", "error_fix", "create"})


def classify(state: str, source: str, review_action: str | None) -> tuple[str, str]:
    """The lifecycle for one object, and how that answer was reached."""
    if review_action is not None:
        if review_action in ENDORSING:
            return "human_confirmed", "derived"
        if review_action in EDITING:
            return "human_edited", "derived"
        if review_action == "reject":
            # A rejection is a human ruling, and the object keeps it: `state` already says rejected, and
            # lifecycle records that a person is the one who said so.
            return "human_confirmed", "derived"

    if source in MACHINE_SOURCES:
        # The gate accepting on confidence is a real event and worth distinguishing from an untouched
        # proposal, because it is the state the badge was silently conflating with human confirmation.
        if state == "auto_accept":
            return "machine_accepted", "inferred"
        return "machine_proposed", "inferred"

    # source says human but no review row exists. That is the agent and 3D-edit gap: something wrote the row
    # as human without leaving an audit trail, and calling it confirmed would invent a ruling nobody made.
    return "machine_proposed", "defaulted"


async def backfill(db, *, apply: bool = False, limit: int | None = None) -> dict:
    """Compute every object's lifecycle. Writes only when apply is true."""
    from sqlalchemy import func, select

    from db.models import Object, Review

    # The latest human action per object, which is what decides a derived answer.
    latest = (
        select(Review.object_id, func.max(Review.ts_ns).label("ts"))
        .group_by(Review.object_id).subquery())
    action_at = (
        select(Review.object_id, Review.action)
        .join(latest, (Review.object_id == latest.c.object_id) & (Review.ts_ns == latest.c.ts))
        .subquery())

    q = (select(Object.object_id, Object.state, Object.source, action_at.c.action)
         .outerjoin(action_at, action_at.c.object_id == Object.object_id))
    if limit:
        q = q.limit(limit)
    rows = (await db.execute(q)).all()

    counts: dict[str, int] = {}
    how: dict[str, int] = {}
    updates: list[tuple] = []
    for object_id, state, source, action in rows:
        lifecycle, basis = classify(state or "", source or "", action)
        counts[lifecycle] = counts.get(lifecycle, 0) + 1
        how[basis] = how.get(basis, 0) + 1
        updates.append((object_id, lifecycle, basis))

    total = len(updates)
    report = {
        "objects": total,
        "by_lifecycle": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_basis": dict(sorted(how.items(), key=lambda kv: -kv[1])),
        "derived_fraction": round(how.get("derived", 0) / total, 5) if total else 0.0,
        "applied": False,
    }

    if not apply:
        log.info("lifecycle_backfill.dry_run", **{k: v for k, v in report.items() if k != "by_lifecycle"})
        return report

    from sqlalchemy import case, update
    CHUNK = 2000
    for i in range(0, total, CHUNK):
        chunk = updates[i:i + CHUNK]
        await db.execute(
            update(Object)
            .where(Object.object_id.in_([u[0] for u in chunk]))
            .values(
                lifecycle=case({u[0]: u[1] for u in chunk}, value=Object.object_id),
                # The basis travels with the value. Without it the column reads as uniformly authoritative
                # when most of it is a default.
                lifecycle_history=case({u[0]: [{"state": u[1], "actor": "backfill", "basis": u[2]}]
                                        for u in chunk}, value=Object.object_id),
            ))
    await db.commit()
    report["applied"] = True
    log.info("lifecycle_backfill.applied", objects=total, **report["by_basis"])
    return report
