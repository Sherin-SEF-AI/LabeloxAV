"""Multi-modal asset and annotation endpoints.

The project spine (LabelProject -> Asset -> Annotation) alongside the AV spine, sharing the same project,
job and issue machinery from labelops.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.assets import store
from services.assets.labelconfig import KINDS, ConfigError, PayloadError, validate_config

router = APIRouter()


def _uid(user) -> str | None:
    return str(user.user_id) if user is not None else None


class AssetsIn(BaseModel):
    project_id: str
    items: list[dict]


class AnnotationIn(BaseModel):
    kind: str
    label: str | None = None
    payload: dict = {}
    fields: dict = {}
    conf: float | None = None
    source: str = "human"
    state: str = "accepted"
    provenance: dict = {}


class AnnotationPatch(BaseModel):
    label: str | None = None
    payload: dict | None = None
    fields: dict | None = None
    state: str | None = None
    expected_version: int | None = None


def _wrap(exc: Exception) -> HTTPException:
    if isinstance(exc, PayloadError | ConfigError):
        return HTTPException(400, str(exc))
    if "moved on" in str(exc):
        return HTTPException(409, str(exc))
    return HTTPException(400, str(exc))


@router.get("/assets/kinds")
async def kinds():
    """Every annotation shape the project spine understands, with what each one means."""
    return {"kinds": [{"kind": k, "description": v} for k, v in KINDS.items()]}


@router.post("/projects/{project_id}/label-config")
async def set_label_config(project_id: str, config: dict = Body(default={}),
                           _user=Depends(require_role("reviewer")),
                           db: AsyncSession = Depends(db_session)):
    """Set the project's labeling interface. Validated here so a broken config cannot be saved and then
    silently reject every annotation later."""
    from uuid import UUID

    from db.models import LabelProject

    p = await db.get(LabelProject, UUID(project_id))
    if p is None:
        raise HTTPException(404, "project not found")
    try:
        p.label_config = validate_config(config)
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from None
    await db.commit()
    return {"project_id": project_id, "label_config": p.label_config}


@router.post("/assets")
async def create_assets(payload: AssetsIn, _user=Depends(require_role("annotator")),
                        db: AsyncSession = Depends(db_session)):
    try:
        return await store.create_assets(db, payload.project_id, payload.items)
    except (store.AssetError, PayloadError) as exc:
        raise _wrap(exc) from None


@router.get("/projects/{project_id}/assets")
async def list_assets(project_id: str, state: str | None = None, media_type: str | None = None,
                      limit: int = 200, offset: int = 0, db: AsyncSession = Depends(db_session)):
    return await store.list_assets(db, project_id, state=state, media_type=media_type,
                                   limit=limit, offset=offset)


@router.get("/projects/{project_id}/stats")
async def project_stats(project_id: str, db: AsyncSession = Depends(db_session)):
    return await store.project_stats(db, project_id)


@router.get("/assets/{asset_id}")
async def get_asset(asset_id: str, db: AsyncSession = Depends(db_session)):
    """The asset, its project's label config, and every annotation on it: one request to open an editor."""
    try:
        return await store.get_asset(db, asset_id)
    except store.AssetError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/assets/{asset_id}/state")
async def set_asset_state(asset_id: str, state: str, _user=Depends(require_role("annotator")),
                          db: AsyncSession = Depends(db_session)):
    try:
        return await store.set_asset_state(db, asset_id, state)
    except store.AssetError as exc:
        raise _wrap(exc) from None


@router.post("/assets/{asset_id}/annotations")
async def create_annotation(asset_id: str, payload: AnnotationIn, user=Depends(require_role("annotator")),
                            db: AsyncSession = Depends(db_session)):
    try:
        return await store.create_annotation(db, asset_id, created_by=_uid(user), **payload.model_dump())
    except (store.AssetError, PayloadError) as exc:
        raise _wrap(exc) from None


@router.get("/assets/{asset_id}/annotations")
async def list_annotations(asset_id: str, kind: str | None = None,
                           db: AsyncSession = Depends(db_session)):
    return {"annotations": await store.list_annotations(db, asset_id, kind)}


@router.patch("/annotations/{annotation_id}")
async def update_annotation(annotation_id: str, payload: AnnotationPatch,
                            _user=Depends(require_role("annotator")),
                            db: AsyncSession = Depends(db_session)):
    try:
        return await store.update_annotation(db, annotation_id, **payload.model_dump())
    except (store.AssetError, PayloadError) as exc:
        raise _wrap(exc) from None


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str, _user=Depends(require_role("annotator")),
                            db: AsyncSession = Depends(db_session)):
    return await store.delete_annotation(db, annotation_id)
