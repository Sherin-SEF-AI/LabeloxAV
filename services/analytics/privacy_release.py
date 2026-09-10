"""The accountant: spending epsilon from a scope's budget, and refusing when it is gone.

`core/privacy.py` holds the mechanisms and knows nothing about who is asking or what they have already
been given. That separation is deliberate, because the mechanism is arithmetic and the budget is policy,
but it means something has to hold the policy, and this is it.

Every release goes through `spend`, which reserves the epsilon before the data is computed. Reserving
first matters: a release that computed its answer and then found the budget empty would either have to
discard work or, much worse, be tempted to return it anyway.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.privacy import DEFAULT_BUDGET, DEFAULT_EPSILON, PrivacyBudgetExhausted

log = get_logger("privacy_release")

# The default scope. One budget for everything released about the corpus, because a per-endpoint budget
# would let an attacker spend the same underlying data several times over by asking different questions.
DEFAULT_SCOPE = "corpus"


async def budget_state(db: AsyncSession, scope: str = DEFAULT_SCOPE) -> dict:
    """What a scope has, what it has spent, and what the log says it spent.

    Both numbers, because they can disagree. The counter is what the accountant enforces; the log sum is
    what actually happened. A reset by hand moves the first and not the second, and the gap between them
    is the record of that reset.
    """
    from db.models import PrivacyBudget, PrivacyReleaseLog

    row = await db.get(PrivacyBudget, scope)
    logged = (await db.execute(
        select(func.coalesce(func.sum(PrivacyReleaseLog.epsilon), 0.0))
        .where(PrivacyReleaseLog.scope == scope))).scalar_one()
    if row is None:
        return {"scope": scope, "exists": False, "epsilon_total": None, "epsilon_spent": 0.0,
                "epsilon_logged": float(logged), "remaining": None,
                "reason": "no budget has been created for this scope"}
    return {"scope": scope, "exists": True, "epsilon_total": row.epsilon_total,
            "epsilon_spent": row.epsilon_spent, "epsilon_logged": float(logged),
            "remaining": round(row.epsilon_total - row.epsilon_spent, 6),
            "window_start": row.window_start.isoformat() if row.window_start else None}


async def spend(db: AsyncSession, *, scope: str = DEFAULT_SCOPE, epsilon: float = DEFAULT_EPSILON,
                endpoint: str, mechanism: str, k: int | None = None,
                cells_released: int = 0, cells_suppressed: int = 0,
                requested_by: str | None = None) -> dict:
    """Reserve epsilon and log the release, or raise when the budget cannot cover it.

    A missing budget is created at the default allowance rather than treated as unlimited. Unlimited is
    the one interpretation that cannot be right: a scope nobody has configured is a scope nobody has
    thought about, and defaulting it to no limit means the first release is unbounded.
    """
    from db.models import PrivacyBudget, PrivacyReleaseLog

    if epsilon <= 0:
        raise ValueError("epsilon must be positive; a zero-epsilon release is an exact release")

    row = await db.get(PrivacyBudget, scope)
    if row is None:
        row = PrivacyBudget(scope=scope, epsilon_total=DEFAULT_BUDGET, epsilon_spent=0.0)
        db.add(row)
        await db.flush()

    remaining = row.epsilon_total - row.epsilon_spent
    if epsilon > remaining:
        raise PrivacyBudgetExhausted(
            f"scope {scope!r} has {remaining:.3f} epsilon left of {row.epsilon_total:.3f} and this "
            f"release needs {epsilon:.3f}; releases compose, so the remaining budget is the remaining "
            f"privacy guarantee and a person has to decide to extend it")

    row.epsilon_spent = round(row.epsilon_spent + epsilon, 6)
    row.updated_at = datetime.now(UTC)
    db.add(PrivacyReleaseLog(scope=scope, endpoint=endpoint, epsilon=epsilon, mechanism=mechanism,
                             k=k, cells_released=cells_released, cells_suppressed=cells_suppressed,
                             requested_by=requested_by))
    await db.commit()
    log.info("privacy.released", scope=scope, endpoint=endpoint, epsilon=epsilon,
             remaining=round(row.epsilon_total - row.epsilon_spent, 6))
    return {"scope": scope, "epsilon": epsilon,
            "remaining": round(row.epsilon_total - row.epsilon_spent, 6),
            "epsilon_total": row.epsilon_total}


async def release_geo_cells(db: AsyncSession, points, *, endpoint: str, scope: str = DEFAULT_SCOPE,
                            epsilon: float = DEFAULT_EPSILON, requested_by: str | None = None) -> dict:
    """Aggregate fixes into released cells and charge the budget for having done so.

    The epsilon is spent whether or not any cell survives suppression. That is not a technicality: asking
    the question is what costs privacy, and a query that returns nothing has still told the asker that
    every cell in that region is thin.
    """
    from core.privacy import aggregate_points

    agg = aggregate_points(points, epsilon=epsilon)
    charged = await spend(db, scope=scope, epsilon=epsilon, endpoint=endpoint,
                          mechanism=agg["mechanism"], k=agg["k_anonymity"],
                          cells_released=agg["n_cells"], cells_suppressed=agg["suppressed_cells"],
                          requested_by=requested_by)
    return {**agg, "budget": charged}
