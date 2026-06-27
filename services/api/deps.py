"""FastAPI dependencies and shared response schemas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends, Header
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_sessionmaker


async def db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def current_user(x_lbx_user_id: str | None = Header(default=None), db: AsyncSession = Depends(db_session)):
    """The acting user, chosen client-side (lightweight, no password). Returns the User row or None.
    Sent as the X-Lbx-User-Id header by the web client."""
    from db.models import User

    if not x_lbx_user_id:
        return None
    try:
        return await db.get(User, UUID(x_lbx_user_id))
    except Exception:  # noqa: BLE001
        return None


class OntologyClassOut(BaseModel):
    id: int
    name: str
    l0: str
    l1: str
    india: bool


class TriageRow(BaseModel):
    object_id: str
    frame_id: str
    session_id: str
    class_id: int
    class_name: str
    conf: float
    state: str
    why: str
    priority: float


class ObjectDetail(BaseModel):
    object_id: str
    frame_id: str
    session_id: str
    track_id: str | None = None
    ts_ns: int
    cam_id: str
    image_url: str
    width: int
    height: int
    class_id: int
    class_name: str
    bbox: list[float]
    mask_polygons: list[list[float]]
    attrs: dict
    conf: float
    state: str
    source: str
    provenance: dict


class ReviewIn(BaseModel):
    reviewer: str = "anon"
    action: str  # confirm | reclassify | adjust_geometry | reject | create
    class_name: str | None = None
    bbox: list[float] | None = None
    attrs: dict | None = None
    state: str | None = None
    time_spent_ms: int = 0


class SegmentIn(BaseModel):
    frame_id: str
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None


class CreateObjectIn(BaseModel):
    class_name: str
    bbox: list[float]                              # xyxy pixel
    attrs: dict = {}
    mask_polygons: list[list[float]] | None = None  # flattened [x,y,x,y,...] per polygon
    state: str = "accepted"


class MaskIn(BaseModel):
    polygons: list[list[float]]                    # flattened [x,y,x,y,...] per polygon
    width: int | None = None
    height: int | None = None


class RelabelTrackIn(BaseModel):
    class_name: str
    state: str = "accepted"


class BulkReviewIn(BaseModel):
    object_ids: list[str]
    action: str = "confirm"            # confirm | accept | reject | reclassify | set_attrs
    class_name: str | None = None      # required for reclassify
    attrs: dict | None = None          # set_attrs: merged into each object's attrs
    state: str | None = None
    reviewer: str = "anon"


class AutolabelStartIn(BaseModel):
    session_id: str
    limit: int | None = None
    compute_target: str = "local"  # local (run here now) | cloud (park for the A100 heavy stack)


class UserCreateIn(BaseModel):
    name: str
    role: str = "annotator"  # admin | reviewer | annotator


class ExportIn(BaseModel):
    name: str = "dataset"
    states: list[str] | None = None
    class_names: list[str] | None = None
    cities: list[str] | None = None
    session_id: str | None = None
    min_conf: float | None = None
    formats: list[str] = ["coco", "parquet"]
    limit: int | None = None


class GoldSealIn(BaseModel):
    name: str = "fleet-v1"
    cities: list[str] | None = None
    session_id: str | None = None
    class_names: list[str] | None = None
    limit: int | None = None


class CalibrateFitIn(BaseModel):
    gold_id: str | None = None
    session_id: str | None = None


class ImportStartIn(BaseModel):
    format: str
    source_uri: str
    target_vehicle: str = "IMPORT-01"
    city: str | None = None
    options: dict = {}


class TrainingStartIn(BaseModel):
    purpose: str = "perception"
    task_type: str = "detection"
    compute_target: str = "local"    # local (RTX 5080) | cloud (RunPod A100)
    dataset_spec: dict = {}          # BuildSpec fields, or {"data_yaml": "..."}
    base_weights: str | None = None
    hparams: dict = {}               # epochs, imgsz, batch
    gate: dict = {}                  # min_map_delta, max_class_drop, min_safe_miou
    promote: bool = False
    notes: str | None = None


class MultipartInitIn(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"


class MultipartSignIn(BaseModel):
    key: str
    upload_id: str
    part_number: int


class MultipartCompleteIn(BaseModel):
    key: str
    upload_id: str
    parts: list[dict]  # [{PartNumber, ETag}]


class MultipartAbortIn(BaseModel):
    key: str
    upload_id: str
