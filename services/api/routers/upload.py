"""Direct-to-storage upload endpoints. The API only signs; the browser PUTs bytes straight to MinIO/S3
(constant API memory, genuinely TB-capable, the literal cloud seam). Multipart for large files, a
single-shot presigned PUT for small ones.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from core.storage import get_object_store
from services.api.deps import (
    MultipartAbortIn,
    MultipartCompleteIn,
    MultipartInitIn,
    MultipartSignIn,
)

router = APIRouter()


def _key(filename: str) -> str:
    safe = filename.replace("/", "_").replace("\\", "_")
    return f"uploads/{uuid.uuid4()}/{safe}"


def _require_upload_key(key: str) -> str:
    """Reject any client-supplied key that is not under uploads/. Without this, sign/complete/abort
    would presign or mutate arbitrary bucket keys (models/, frames/, masks/) -- an object-store IDOR."""
    if not isinstance(key, str) or not key.startswith("uploads/") or ".." in key:
        raise HTTPException(400, "key must be an uploads/ object created via /upload/init")
    return key


@router.post("/upload/init")
async def init(payload: MultipartInitIn):
    store = get_object_store()
    store.ensure_bucket()
    key = _key(payload.filename)
    upload_id = store.create_multipart(key, payload.content_type)
    return {"key": key, "upload_id": upload_id}


@router.post("/upload/sign")
async def sign(payload: MultipartSignIn):
    url = get_object_store().presign_part(_require_upload_key(payload.key), payload.upload_id, payload.part_number)
    return {"url": url, "part_number": payload.part_number}


@router.post("/upload/complete")
async def complete(payload: MultipartCompleteIn):
    uri = get_object_store().complete_multipart(_require_upload_key(payload.key), payload.upload_id, payload.parts)
    return {"uri": uri, "key": payload.key}


@router.post("/upload/abort")
async def abort(payload: MultipartAbortIn):
    get_object_store().abort_multipart(_require_upload_key(payload.key), payload.upload_id)
    return {"ok": True}


@router.post("/upload/presign-put")
async def presign_put(payload: MultipartInitIn):
    store = get_object_store()
    store.ensure_bucket()
    key = _key(payload.filename)
    url = store.presigned_put(key, payload.content_type)
    return {"key": key, "url": url, "uri": store.uri(key)}
