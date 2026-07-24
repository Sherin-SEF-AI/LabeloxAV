"""Assets and their annotations: create, import, read, and write with validation.

Every write goes through services/assets/labelconfig.py, so a malformed span or a label the project never
declared is refused at the door rather than discovered later in an export.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Annotation, Asset, LabelProject
from services.assets.labelconfig import (
    PayloadError,
    check_label_allowed,
    validate_fields,
    validate_payload,
)

log = get_logger("assets_store")

MEDIA_TYPES = ("image", "video", "audio", "text", "timeseries", "document", "pointcloud", "dialogue")


class AssetError(RuntimeError):
    pass


def _asset_dict(a: Asset) -> dict:
    return {"asset_id": str(a.asset_id), "project_id": str(a.project_id), "media_type": a.media_type,
            "uri": a.uri, "text": a.text, "external_id": a.external_id,
            "frame_id": str(a.frame_id) if a.frame_id else None,
            "session_id": str(a.session_id) if a.session_id else None,
            "meta": a.meta or {}, "state": a.state,
            "created_at": a.created_at.isoformat() if a.created_at else None}


def _ann_dict(a: Annotation) -> dict:
    return {"annotation_id": str(a.annotation_id), "asset_id": str(a.asset_id), "kind": a.kind,
            "label": a.label, "payload": a.payload or {}, "fields": a.fields or {},
            "conf": a.conf, "source": a.source, "state": a.state, "version": a.version,
            "provenance": a.provenance or {},
            "created_at": a.created_at.isoformat() if a.created_at else None}


async def create_assets(db: AsyncSession, project_id: str, items: list[dict]) -> dict:
    """Bulk-create assets. `external_id` makes import idempotent: re-importing the same source updates the
    existing row instead of producing a duplicate, which matters because import is the operation people retry.
    """
    project = await db.get(LabelProject, UUID(project_id))
    if project is None:
        raise AssetError("project not found")
    if not items:
        return {"created": 0, "updated": 0, "assets": []}

    existing = {}
    ext_ids = [str(i["external_id"]) for i in items if i.get("external_id")]
    if ext_ids:
        rows = (await db.execute(select(Asset).where(
            Asset.project_id == project.project_id, Asset.external_id.in_(ext_ids)))).scalars().all()
        existing = {a.external_id: a for a in rows}

    created = updated = 0
    out: list[Asset] = []
    for item in items:
        mt = item.get("media_type") or project.modality or "image"
        if mt not in MEDIA_TYPES:
            raise AssetError(f"media_type must be one of {MEDIA_TYPES}")
        if not item.get("uri") and not item.get("text") and not item.get("frame_id"):
            raise AssetError("an asset needs a uri, inline text, or a frame_id")
        ext = str(item["external_id"]) if item.get("external_id") else None
        a = existing.get(ext) if ext else None
        if a is None:
            a = Asset(project_id=project.project_id, media_type=mt, external_id=ext)
            db.add(a)
            created += 1
        else:
            updated += 1
        a.uri = item.get("uri", a.uri)
        a.text = item.get("text", a.text)
        a.meta = item.get("meta", a.meta) or {}
        if item.get("frame_id"):
            a.frame_id = UUID(str(item["frame_id"]))
        if item.get("session_id"):
            a.session_id = UUID(str(item["session_id"]))
        out.append(a)

    await db.commit()
    log.info("assets.created", project=project_id, created=created, updated=updated)
    return {"created": created, "updated": updated, "assets": [_asset_dict(a) for a in out]}


async def list_assets(db: AsyncSession, project_id: str, *, state: str | None = None,
                      media_type: str | None = None, limit: int = 200, offset: int = 0) -> dict:
    stmt = select(Asset).where(Asset.project_id == UUID(project_id))
    if state:
        stmt = stmt.where(Asset.state == state)
    if media_type:
        stmt = stmt.where(Asset.media_type == media_type)
    total = (await db.execute(
        select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (await db.execute(
        stmt.order_by(Asset.created_at).offset(offset).limit(limit))).scalars().all()
    return {"total": int(total), "assets": [_asset_dict(a) for a in rows]}


async def get_asset(db: AsyncSession, asset_id: str) -> dict:
    a = await db.get(Asset, UUID(asset_id))
    if a is None:
        raise AssetError("asset not found")
    anns = (await db.execute(
        select(Annotation).where(Annotation.asset_id == a.asset_id)
        .order_by(Annotation.created_at))).scalars().all()
    project = await db.get(LabelProject, a.project_id)
    return {**_asset_dict(a),
            "label_config": (project.label_config or {}) if project else {},
            "annotations": [_ann_dict(x) for x in anns]}


async def set_asset_state(db: AsyncSession, asset_id: str, state: str) -> dict:
    if state not in ("new", "in_progress", "labeled", "skipped"):
        raise AssetError("invalid asset state")
    a = await db.get(Asset, UUID(asset_id))
    if a is None:
        raise AssetError("asset not found")
    a.state = state
    await db.commit()
    if state == "labeled":
        from services.integrations.webhooks import emit

        await emit("asset.labeled", {"asset_id": asset_id, "media_type": a.media_type},
                   project_id=str(a.project_id))
    return _asset_dict(a)


async def create_annotation(db: AsyncSession, asset_id: str, *, kind: str, label: str | None = None,
                            payload: dict | None = None, fields: dict | None = None,
                            conf: float | None = None, source: str = "human",
                            state: str = "accepted", provenance: dict | None = None,
                            created_by: str | None = None) -> dict:
    """Write one annotation, validated against its kind and the project's label config."""
    asset = await db.get(Asset, UUID(asset_id))
    if asset is None:
        raise AssetError("asset not found")
    project = await db.get(LabelProject, asset.project_id)
    cfg = (project.label_config or {}) if project else {}

    payload = validate_payload(kind, payload)
    check_label_allowed(cfg, label, kind)
    fields = validate_fields(cfg, fields)

    # A span must actually lie inside the text it annotates; an out-of-range span silently produces garbage
    # on export, so it is refused here where the asset body is at hand.
    if kind == "span":
        body = asset.text or ""
        if payload["end"] > len(body):
            raise PayloadError(f"span end {payload['end']} is past the end of the text ({len(body)} chars)")
        payload.setdefault("quote", body[payload["start"]:payload["end"]])

    ann = Annotation(asset_id=asset.asset_id, kind=kind, label=label, payload=payload,
                     fields=fields, conf=conf, source=source, state=state,
                     provenance=provenance or {},
                     created_by=UUID(created_by) if created_by else None)
    db.add(ann)
    if asset.state == "new":
        asset.state = "in_progress"
    await db.commit()
    from services.integrations.webhooks import emit

    await emit("annotation.created",
               {"annotation_id": str(ann.annotation_id), "asset_id": asset_id, "kind": kind,
                "label": label},
               project_id=str(asset.project_id))
    return _ann_dict(ann)


async def update_annotation(db: AsyncSession, annotation_id: str, *, label: str | None = None,
                            payload: dict | None = None, fields: dict | None = None,
                            state: str | None = None, expected_version: int | None = None) -> dict:
    ann = await db.get(Annotation, UUID(annotation_id))
    if ann is None:
        raise AssetError("annotation not found")
    if expected_version is not None and ann.version != expected_version:
        raise AssetError(f"annotation moved on (version {ann.version}, expected {expected_version})")

    asset = await db.get(Asset, ann.asset_id)
    project = await db.get(LabelProject, asset.project_id) if asset else None
    cfg = (project.label_config or {}) if project else {}

    if payload is not None:
        ann.payload = validate_payload(ann.kind, payload)
    if label is not None:
        check_label_allowed(cfg, label, ann.kind)
        ann.label = label
    if fields is not None:
        ann.fields = validate_fields(cfg, fields)
    if state is not None:
        ann.state = state
    ann.version += 1
    await db.commit()
    return _ann_dict(ann)


async def delete_annotation(db: AsyncSession, annotation_id: str) -> dict:
    n = (await db.execute(
        delete(Annotation).where(Annotation.annotation_id == UUID(annotation_id)))).rowcount
    await db.commit()
    return {"deleted": bool(n)}


async def list_annotations(db: AsyncSession, asset_id: str, kind: str | None = None) -> list[dict]:
    stmt = select(Annotation).where(Annotation.asset_id == UUID(asset_id))
    if kind:
        stmt = stmt.where(Annotation.kind == kind)
    rows = (await db.execute(stmt.order_by(Annotation.created_at))).scalars().all()
    return [_ann_dict(a) for a in rows]


async def project_stats(db: AsyncSession, project_id: str) -> dict:
    """Assets by state and annotations by kind: the progress readout for a multi-modal project."""
    by_state = dict((await db.execute(
        select(Asset.state, func.count()).where(Asset.project_id == UUID(project_id))
        .group_by(Asset.state))).all())
    by_kind = dict((await db.execute(
        select(Annotation.kind, func.count())
        .join(Asset, Asset.asset_id == Annotation.asset_id)
        .where(Asset.project_id == UUID(project_id))
        .group_by(Annotation.kind))).all())
    return {"project_id": project_id,
            "assets_by_state": {k: int(v) for k, v in by_state.items()},
            "annotations_by_kind": {k: int(v) for k, v in by_kind.items()},
            "total_assets": int(sum(by_state.values())),
            "total_annotations": int(sum(by_kind.values()))}
