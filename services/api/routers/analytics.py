"""Analytics and eval dashboard endpoints (the sales sheet). Read-only aggregates over Postgres.

These thin handlers delegate to services.analytics.dashboards, which open their own sessions, so
no db dependency is needed here. Mounted at /api by main.py.
"""

from __future__ import annotations

from fastapi import APIRouter

from services.analytics import dashboards

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
async def geo(session_id: str | None = None, limit: int = 2000):
    return await dashboards.geo_points(session_id, limit)


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
