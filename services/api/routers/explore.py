"""Explore workspace endpoints: the embeddings map, faceted counts, bulk tagging, and saved views.

The predicate posted to these endpoints is the same shape a CurationSlice stores, so a filter built by
clicking facets can be saved as a cohort and exported without redefining it (services/curation/slices.py).
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CurationSlice
from services.api.deps import db_session, require_role
from services.curation import projection as proj_svc
from services.curation.slices import create_slice, slice_to_export_spec
from services.explore import facets as facet_svc
from services.explore import tags as tag_svc
from services.explore.query import frame_select, object_select

router = APIRouter()


class ProjectionIn(BaseModel):
    kind: str = "object"          # object | frame
    space: str = "dino"           # dino | siglip (frame only)
    session_id: str | None = None
    method: str = "umap"          # umap | pca
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42
    min_cluster_size: int = 15
    limit: int = proj_svc.MAX_POINTS
    notes: str | None = None


class TagIn(BaseModel):
    level: str = "object"         # object | frame
    predicate: dict = {}
    add: list[str] = []
    remove: list[str] = []


class ViewIn(BaseModel):
    name: str
    predicate: dict = {}
    description: str | None = None


# ---- embeddings map -------------------------------------------------------------------------------------

@router.post("/explore/projection")
async def fit_projection(payload: ProjectionIn, _user=Depends(require_role("reviewer")),
                         db: AsyncSession = Depends(db_session)):
    """Fit and persist a 2D map of an embedding space. Reviewer-gated: a fit over the full corpus is minutes
    of CPU, so it is not an anonymous annotator action."""
    try:
        return await proj_svc.fit_projection(
            db, kind=payload.kind, space=payload.space, session_id=payload.session_id,
            method=payload.method, n_neighbors=payload.n_neighbors, min_dist=payload.min_dist,
            seed=payload.seed, min_cluster_size=payload.min_cluster_size, limit=payload.limit,
            notes=payload.notes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/explore/projections")
async def list_projections(limit: int = 50, db: AsyncSession = Depends(db_session)):
    return {"projections": await proj_svc.list_projections(db, limit)}


@router.get("/explore/projection/{projection_id}/points")
async def projection_points(projection_id: str, limit: int = proj_svc.MAX_POINTS,
                            db: AsyncSession = Depends(db_session)):
    return await proj_svc.projection_points(db, projection_id, limit)


@router.delete("/explore/projection/{projection_id}")
async def delete_projection(projection_id: str, _user=Depends(require_role("reviewer")),
                            db: AsyncSession = Depends(db_session)):
    return await proj_svc.delete_projection(db, projection_id)


# ---- facets + selection ---------------------------------------------------------------------------------

@router.post("/explore/facets")
async def facets(predicate: dict = Body(default={}), db: AsyncSession = Depends(db_session)):
    """Faceted counts under the current predicate, each facet computed with its own clause dropped."""
    return await facet_svc.object_facets(db, predicate)


@router.post("/explore/select")
async def select_ids(predicate: dict = Body(default={}), level: str = "object", limit: int = 5000,
                     db: AsyncSession = Depends(db_session)):
    """Resolve a predicate to concrete ids and a count: what a lasso or a filter currently covers."""
    from sqlalchemy import func

    from db.models import Frame, Object

    if level == "frame":
        count = (await db.execute(frame_select(predicate, func.count()))).scalar_one()
        rows = (await db.execute(frame_select(predicate, Frame.frame_id).limit(limit))).scalars().all()
    else:
        count = (await db.execute(object_select(predicate, func.count()))).scalar_one()
        rows = (await db.execute(object_select(predicate, Object.object_id).limit(limit))).scalars().all()
    return {"level": level, "count": int(count), "ids": [str(r) for r in rows]}


# ---- tags -----------------------------------------------------------------------------------------------

@router.post("/explore/tag")
async def apply_tags(payload: TagIn, _user=Depends(require_role("annotator")),
                     db: AsyncSession = Depends(db_session)):
    """Add and/or remove curation tags across everything matching the predicate. Idempotent."""
    try:
        return await tag_svc.apply_tags(db, level=payload.level, pred=payload.predicate,
                                        add=payload.add, remove=payload.remove)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/explore/tags")
async def tag_vocabulary(level: str = "object", db: AsyncSession = Depends(db_session)):
    return {"level": level, "tags": await tag_svc.tag_vocabulary(db, level)}


# ---- saved views ----------------------------------------------------------------------------------------

@router.get("/explore/views")
async def list_views(db: AsyncSession = Depends(db_session)):
    """Saved views are CurationSlice rows: the explorer and the export path share one cohort definition."""
    rows = (await db.execute(
        select(CurationSlice).order_by(CurationSlice.created_at.desc()).limit(200))).scalars().all()
    return {"views": [{"slice_id": str(r.slice_id), "name": r.name, "predicate": r.predicate,
                       "description": r.description,
                       "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]}


@router.post("/explore/views")
async def save_view(payload: ViewIn, _user=Depends(require_role("annotator"))):
    return await create_slice(payload.name, payload.predicate, payload.description)


@router.get("/explore/views/{slice_id}/export-spec")
async def view_export_spec(slice_id: str, db: AsyncSession = Depends(db_session)):
    """Turn a saved view into the export SliceSpec, so a curated cohort exports without being redefined."""
    from uuid import UUID

    row = await db.get(CurationSlice, UUID(slice_id))
    if row is None:
        raise HTTPException(404, "view not found")
    return slice_to_export_spec(row)
