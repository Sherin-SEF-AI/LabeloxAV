"""FastAPI dependencies and shared response schemas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_sessionmaker

# Role hierarchy: a higher rank satisfies any floor at or below it. admin is the superuser.
ROLE_RANK = {"annotator": 1, "reviewer": 2, "admin": 3}


def role_rank(role: str | None) -> int:
    return ROLE_RANK.get(role or "", 0)


async def db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def current_user(x_lbx_user_id: str | None = Header(default=None), db: AsyncSession = Depends(db_session)):
    """The acting user, chosen client-side (lightweight, no password). Returns the User row or None.
    Sent as the X-Lbx-User-Id header by the web client. Open (read) endpoints may receive None."""
    from db.models import User

    if not x_lbx_user_id:
        return None
    try:
        return await db.get(User, UUID(x_lbx_user_id))
    except Exception:  # noqa: BLE001
        return None


async def require_user(user=Depends(current_user)):
    """Dependency that rejects unauthenticated callers (401). Use on any state-changing route."""
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required (X-Lbx-User-Id)")
    return user


def require_role(min_role: str):
    """Dependency factory: require the acting user to hold at least `min_role` (else 403)."""

    async def _dep(user=Depends(require_user)):
        if role_rank(user.role) < role_rank(min_role):
            raise HTTPException(status_code=403, detail=f"requires {min_role} role or higher")
        return user

    return _dep


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
    version: int = 1
    rot_deg: float = 0.0
    keypoints: dict | None = None
    polyline: list[list[float]] | None = None


class ReviewIn(BaseModel):
    reviewer: str = "anon"
    action: str  # confirm | reclassify | adjust_geometry | reject | create
    class_name: str | None = None
    bbox: list[float] | None = None
    attrs: dict | None = None
    state: str | None = None
    time_spent_ms: int = 0
    expected_version: int | None = None  # optimistic lock: 409 if the object moved on under the editor
    rot_deg: float | None = None         # oriented-box rotation (only updated when provided)
    keypoints: dict | None = None        # keypoints/skeleton (only updated when provided)
    mask_polygons: list[list[float]] | None = None  # write the mask in the same request (atomic save)
    polyline: list[list[float]] | None = None       # open polyline points (only updated when provided)


class SegmentIn(BaseModel):
    frame_id: str
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None


class CreateObjectIn(BaseModel):
    class_name: str
    bbox: list[float]                              # xyxy pixel (axis-aligned AABB)
    attrs: dict = {}
    mask_polygons: list[list[float]] | None = None  # flattened [x,y,x,y,...] per polygon
    state: str = "accepted"
    idem_key: str | None = None                    # client temp id; de-dupes a retried/raced create
    rot_deg: float = 0.0                           # oriented-box rotation about the box centre
    keypoints: dict | None = None                  # {"skeleton": str, "points": [[x,y,v],...]} image px
    polyline: list[list[float]] | None = None      # open polyline points [[x,y],...] for linear features


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
