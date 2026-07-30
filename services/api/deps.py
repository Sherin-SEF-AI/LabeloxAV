"""FastAPI dependencies and shared response schemas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.session import get_sessionmaker
from services.api.auth_token import bearer_payload

# Role hierarchy: a higher rank satisfies any floor at or below it. admin is the superuser.
ROLE_RANK = {"annotator": 1, "reviewer": 2, "admin": 3}


def role_rank(role: str | None) -> int:
    return ROLE_RANK.get(role or "", 0)


async def db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


def _media_cookie_payload(request: Request, settings):
    """The media-cookie payload, but only on an image subresource and only if the token is media-scoped.

    Deliberately narrow. The cookie is attached by the browser to every /api request it makes, so honouring
    it anywhere else would turn an automatically-sent credential into a general-purpose one.
    """
    from services.api.auth_token import MEDIA_SCOPE, verify_token
    from services.api.media import MEDIA_COOKIE, is_media_read

    if request.method not in ("GET", "HEAD") or not is_media_read(request.url.path):
        return None
    cookie = request.cookies.get(MEDIA_COOKIE)
    payload = verify_token(cookie, settings.auth.signing_key) if cookie else None
    return payload if (payload is not None and payload.scope == MEDIA_SCOPE) else None


async def current_user(
    request: Request,
    authorization: str | None = Header(default=None),
    x_lbx_user_id: str | None = Header(default=None),
    db: AsyncSession = Depends(db_session),
):
    """The acting user, identified by a signed Bearer token. Returns the User row or None. Verifies the token
    signature and expiry, then enforces revocation: a v2 token whose token_version does not match the user row
    is rejected. The legacy plaintext X-Lbx-User-Id header is honored ONLY when auth is disabled (unit tests).
    Open (read) endpoints may receive None."""
    from db.models import User

    settings = get_settings()
    payload = bearer_payload(authorization, settings.auth.signing_key,
                             accept_legacy=settings.auth.accept_legacy_tokens)
    if payload is None:
        # The media cookie, honoured only for an image subresource. Some of those routes sit under a router
        # with its own role floor, so resolving identity here as well is what keeps the cookie working on
        # them; without it the middleware would admit the request and the dependency would still refuse it.
        payload = _media_cookie_payload(request, settings)
    if payload is not None:
        uid = payload.uid
    elif not settings.auth.enabled and x_lbx_user_id:
        uid = x_lbx_user_id
    else:
        return None
    try:
        user = await db.get(User, UUID(uid))
    except Exception:  # noqa: BLE001
        return None
    if user is None:
        return None
    # Revocation: a v2 token must carry the user's current token_version. Legacy tokens (tv None) and the
    # auth-disabled header path carry no version and skip this check.
    if payload is not None and not payload.legacy and payload.token_version != user.token_version:
        return None
    return user


async def require_user(user=Depends(current_user)):
    """Dependency that rejects unauthenticated callers (401). Use on any state-changing route."""
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required (Bearer token)")
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


class TriageFlag(BaseModel):
    """One reason an object is in the queue, with how loudly it should read.

    Structured rather than folded into the prose on `why`, so severity is decided where the evidence is and
    a client renders it instead of recovering it with regexes over a joined string.
    """

    code: str                                   # mask_box | class_conflict | rare_class | low_conf | review_band
    label: str                                  # what the reader is asked to check, in words
    severity: str                               # high | medium | low


class TriageRow(BaseModel):
    object_id: str
    frame_id: str
    session_id: str
    class_id: int
    class_name: str
    conf: float
    state: str
    why: str                                    # the same reasons as prose, kept for existing readers
    flags: list[TriageFlag] = []                # worst first, so ordering does not need re-deriving
    priority: float
    source: str = "human"                       # imported | fused | human | auto_accept | interpolated
    import_format: str | None = None            # mapillary | idd | bdd | ... when source == imported


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
    cuboid_3d: dict | None = None
    # Sign typing and road text, read-only. Served so a reviewer can see what the classifier decided; a
    # wrong type is corrected by re-running recognition, not by editing the field.
    sign_type: str | None = None
    sign_category: str | None = None
    ocr_text: str | None = None
    ocr_lang: str | None = None
    ocr_conf: float | None = None


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
    cuboid_3d: dict | None = None                    # ego-frame {center,size,yaw} (only when provided)


class SegmentIn(BaseModel):
    frame_id: str
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None
    precise: bool = False  # panoptic mode: follow the visible edge tightly and keep occlusion holes


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
    cuboid_3d: dict | None = None                  # ego-frame {center:[x,y,z], size:[w,l,h], yaw:rad}


class MaskIn(BaseModel):
    polygons: list[list[float]]                    # flattened [x,y,x,y,...] per polygon
    width: int | None = None
    height: int | None = None


class RelateIn(BaseModel):
    to_object_id: str                              # the target object this one relates to
    kind: str                                      # rider_of|towed_by|part_of|member_of|occludes


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
    vehicle_ids: list[str] | None = None
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
