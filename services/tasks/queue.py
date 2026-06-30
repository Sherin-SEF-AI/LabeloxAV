"""Milestone I: the annotation task queue. The Assignment model and a manual one-at-a-time assign already
exist (versioning.collaborate), but the queue mechanics did not: bulk enqueue of a work list (e.g. from a
curation slice or the active-learning ranking), a concurrency-safe claim of the next pending task so two
annotators never grab the same item, a validated status state machine, and queue depth. The transition rules
are a pure state machine, tested without infra; claim_next uses a row lock with skip-locked so concurrent
claims fall through to different rows.

Status flow: assigned -> in_progress -> submitted -> done, with in_progress <-> assigned (release) and
submitted -> in_progress (reviewer sends back). done is terminal. Any other move is refused (fail-closed).
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("task_queue")

_TRANSITIONS: dict[str, set[str]] = {
    "assigned": {"in_progress"},
    "in_progress": {"submitted", "assigned"},
    "submitted": {"done", "in_progress"},
    "done": set(),
}


def valid_transition(cur: str, to: str) -> bool:
    """Whether moving a task from cur to to is allowed by the state machine."""
    return to in _TRANSITIONS.get(cur, set())


async def enqueue_tasks(item_ids: list, user_id) -> dict:
    """Bulk-create assigned tasks for a user from a work list (slice members, ranked candidates, etc.)."""
    from db.models import Assignment
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        for it in item_ids:
            db.add(Assignment(item_id=str(it), user_id=user_id, status="assigned"))
        await db.commit()
    log.info("task_queue.enqueue", user=str(user_id), n=len(item_ids))
    return {"enqueued": len(item_ids), "user_id": str(user_id)}


async def claim_next(user_id) -> dict:
    """Atomically claim the user's oldest still-assigned task, moving it to in_progress. The row lock with
    skip_locked means a concurrent claim skips this row instead of blocking or double-claiming."""
    from sqlalchemy import select

    from db.models import Assignment
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        row = (await db.execute(
            select(Assignment).where(Assignment.user_id == user_id, Assignment.status == "assigned")
            .order_by(Assignment.created_at).limit(1).with_for_update(skip_locked=True))).scalar_one_or_none()
        if row is None:
            return {"claimed": None}
        row.status = "in_progress"
        await db.commit()
        claimed = {"claimed": str(row.assignment_id), "item_id": row.item_id}
    log.info("task_queue.claim", user=str(user_id), assignment=claimed["claimed"])
    return claimed


async def advance_task(assignment_id, to_status: str) -> dict:
    """Move a task to to_status if the transition is valid, else refuse (fail-closed)."""
    from db.models import Assignment
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        a = await db.get(Assignment, assignment_id)
        if a is None:
            return {"error": "assignment not found"}
        if not valid_transition(a.status, to_status):
            return {"invalid_transition": True, "from": a.status, "to": to_status}
        a.status = to_status
        await db.commit()
    return {"assignment_id": str(assignment_id), "status": to_status}


async def queue_stats(user_id) -> dict:
    """Task counts per status for a user, so the queue depth and work-in-progress are visible."""
    from sqlalchemy import func, select

    from db.models import Assignment
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Assignment.status, func.count()).where(Assignment.user_id == user_id)
            .group_by(Assignment.status))).all()
    counts = {status: int(n) for status, n in rows}
    return {"user_id": str(user_id), "counts": counts, "total": sum(counts.values())}
