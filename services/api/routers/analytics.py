"""Analytics and eval dashboard endpoints (the sales sheet). Read-only aggregates over Postgres.

These thin handlers delegate to services.analytics.dashboards, which open their own sessions, so
no db dependency is needed here. Mounted at /api by main.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from services.analytics import dashboards
from services.api.deps import current_user, require_role

router = APIRouter()


@router.get("/analytics/overview")
async def overview(session_id: str | None = None):
    return await dashboards.overview(session_id)


@router.get("/analytics/classes")
async def classes(session_id: str | None = None):
    return await dashboards.class_distribution(session_id)


@router.get("/analytics/source-mix")
async def source_mix(session_id: str | None = None):
    return await dashboards.label_source_mix(session_id)


@router.get("/analytics/scenarios")
async def scenarios(session_id: str | None = None):
    return await dashboards.scenario_coverage(session_id)


@router.get("/analytics/geo")
async def geo(session_id: str | None = None, limit: int = 20000, epsilon: float | None = None,
              user=Depends(current_user)):
    """Where the corpus was collected, as aggregated cells. Never as raw fixes.

    This returned latitude and longitude pairs straight from `Frame.gnss`. A driving trace is among the
    most identifying data a vehicle produces: a handful of points reconstructs a home address, a workplace
    and a daily route, and removing names changes none of that. It is also exactly what a buyer
    legitimately wants in aggregate.

    So the fixes become 250 metre cells, cells holding fewer than ten fixes are dropped rather than
    noised, and what survives gets Laplace noise charged against the scope's epsilon budget. The response
    says how much was suppressed, because a sparse map and a fleet that did not drive there are different
    facts.
    """
    from core.privacy import DEFAULT_EPSILON, PrivacyBudgetExhausted
    from db.session import get_sessionmaker
    from services.analytics.privacy_release import release_geo_cells

    points = await dashboards.geo_points(session_id, limit)
    raw = [(p.get("lat"), p.get("lon")) for p in points]
    async with get_sessionmaker()() as db:
        try:
            return await release_geo_cells(
                db, raw, endpoint="/analytics/geo",
                epsilon=float(epsilon or DEFAULT_EPSILON),
                requested_by=getattr(user, "name", None) or getattr(user, "email", None))
        except PrivacyBudgetExhausted as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/analytics/privacy-budget", dependencies=[Depends(require_role("reviewer"))])
async def privacy_budget():
    """What the corpus scope has spent, by the counter and by the log.

    Both, because they can disagree: the counter is what the accountant enforces and the log sum is what
    actually happened, and a budget reset by hand moves the first and not the second.
    """
    from db.session import get_sessionmaker
    from services.analytics.privacy_release import budget_state

    async with get_sessionmaker()() as db:
        return await budget_state(db)


@router.get("/analytics/review-agreement")
async def review_agreement():
    return await dashboards.review_agreement()


@router.get("/analytics/pii")
async def pii(session_id: str | None = None):
    return await dashboards.pii_coverage(session_id)


# ---- Data Intelligence Layer (M1.7) ----
@router.get("/analytics/scene-splits")
async def scene_splits(session_id: str | None = None):
    return await dashboards.scene_splits(session_id)


@router.get("/analytics/dedup-rate")
async def dedup_rate(session_id: str | None = None):
    return await dashboards.dedup_rate(session_id)


@router.get("/analytics/growth")
async def growth():
    return await dashboards.dataset_growth()


@router.get("/analytics/cluster-map")
async def cluster_map(limit: int = 1500):
    return await dashboards.cluster_map(limit)


@router.get("/analytics/report")
async def report(session_id: str | None = None):
    """Consolidated summary (the buyer quality sheet), bundled for export."""
    return {
        "overview": await dashboards.overview(session_id),
        "classes": await dashboards.class_distribution(session_id),
        "source_mix": await dashboards.label_source_mix(session_id),
        "scene_splits": await dashboards.scene_splits(session_id),
        "dedup_rate": await dashboards.dedup_rate(session_id),
        "scenarios": await dashboards.scenario_coverage(session_id),
        "pii": await dashboards.pii_coverage(session_id),
    }


@router.get("/analytics/productivity")
async def productivity():
    """M-F.4: the DataOps operations view (per-reviewer throughput/correction/agreement, cost, trend)."""
    from services.analytics.productivity import productivity_report

    return await productivity_report()


@router.get("/analytics/label-value", dependencies=[Depends(require_role("annotator"))])
async def label_value(run_id: str | None = None):
    """What the next label of each class the gate is short on is worth, per rupee.

    Rows the inputs cannot price come back with the reason rather than a number, so a class nobody has
    timed is visibly unpriced instead of ranking last on a fabricated median.
    """
    from db.session import get_sessionmaker
    from services.analytics.label_value import marginal_value

    # Its own session, like every other handler in this router: they delegate to modules that open one,
    # which is why this file has never carried a db dependency.
    async with get_sessionmaker()() as db:
        return await marginal_value(db, run_id=run_id)


@router.post("/analytics/label-value/snapshot", dependencies=[Depends(require_role("reviewer"))])
async def label_value_snapshot(run_id: str | None = None):
    """Record the ranking now, so a decision made today stays explainable against today's inputs."""
    from db.session import get_sessionmaker
    from services.analytics.label_value import snapshot

    async with get_sessionmaker()() as db:
        return await snapshot(db, run_id=run_id)
