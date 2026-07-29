"""LabeloxSec API: the security-domain surface (ANPR reads, the plate watchlist, static-camera sessions).

The Sec pack shipped with an ontology, a static-camera scene model, a tested India plate-format kernel, and a
recogniser, and no endpoint exposed any of it. Nothing outside the Python package could reach the second
domain, which is why it had no product.

Every route here is capability-gated, not role-gated alone. Reading a registration mark is lawful for an
authorised security deployment and is exactly what the AV pack must never do, since under DPDPA a plate is
personal data that the privacy plane blurs. The gate lives in the pack, so the same binary refuses in one
domain and permits in the other.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.anpr.recognize import AnprNotAuthorised
from services.api.deps import db_session, require_role

log = get_logger("api_security")

# Reviewer floor: plate reads and the watchlist are personal data and operational security state. An
# annotator drawing boxes has no reason to enumerate who drove past a camera.
router = APIRouter(dependencies=[Depends(require_role("reviewer"))])


def _refuse(exc: AnprNotAuthorised) -> HTTPException:
    # 403, not 400: this is an authorisation decision about the domain, not a malformed request.
    return HTTPException(status_code=403, detail=str(exc))


# ---------------------------------------------------------------- pack

@router.get("/security/pack")
async def security_pack(pack_id: str = "sec"):
    """What the security pack authorises and how it sees the world.

    The console reads this to decide what to render: a deployment running the AV pack should be told ANPR is
    refused there, not shown a plate console that 403s on every action.
    """
    from packs.registry import get_pack, pack_ids

    try:
        pack = get_pack(pack_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"unknown pack {pack_id!r}") from exc

    m = pack.manifest
    return {
        "pack_id": pack_id,
        "name": getattr(m, "name", pack_id),
        "capabilities": sorted(m.capabilities),
        "anpr_authorised": "anpr" in m.capabilities,
        "static_camera": "static_camera" in m.capabilities,
        "safety_classes": sorted(pack.safety_policy.critical_class_names()),
        "available_packs": sorted(pack_ids()),
    }


@router.get("/security/sessions")
async def security_sessions(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                            db: AsyncSession = Depends(db_session)):
    """Sessions captured under the security pack.

    This is per-session pack routing made visible: a session records the pack it belongs to, and the console
    lists only its own domain's captures instead of every dashcam drive in the corpus.
    """
    from sqlalchemy import func

    from db.models import Frame
    from db.models import PlateRead as PlateReadRow
    from db.models import Session as DbSession

    rows = (await db.execute(
        select(DbSession).where(DbSession.pack_id == "sec")
        .order_by(DbSession.start_ts_ns.desc()).offset(offset).limit(limit))).scalars().all()
    total = (await db.execute(
        select(func.count()).select_from(DbSession).where(DbSession.pack_id == "sec"))).scalar_one()

    ids = [r.session_id for r in rows]
    reads_by_session: dict = {}
    cams: dict = {}
    if ids:
        reads_by_session = {sid: int(n) for sid, n in (await db.execute(
            select(PlateReadRow.session_id, func.count())
            .where(PlateReadRow.session_id.in_(ids))
            .group_by(PlateReadRow.session_id))).all()}
        # The camera comes from the frames, not from vehicle_id. A static-camera session has no ego vehicle
        # by design, which is the whole point of the Sec fork, so the vehicle column is null for every row
        # this endpoint exists to describe.
        for sid, cam in (await db.execute(
                select(Frame.session_id, Frame.cam_id).where(Frame.session_id.in_(ids)).distinct())).all():
            cams.setdefault(sid, cam)

    return {"total": int(total), "offset": offset, "limit": limit, "sessions": [
        {"session_id": str(r.session_id), "camera_id": cams.get(r.session_id), "city": r.city,
         "start_ts_ns": r.start_ts_ns, "pack_id": r.pack_id,
         "plate_reads": int(reads_by_session.get(r.session_id, 0))}
        for r in rows]}


# ---------------------------------------------------------------- watchlist

class WatchlistIn(BaseModel):
    plate: str
    reason: str | None = None
    severity: str = "warn"          # info | warn | critical


@router.get("/security/watchlist")
async def get_watchlist(active_only: bool = True, limit: int = Query(500, ge=1, le=2000),
                        db: AsyncSession = Depends(db_session)):
    from services.anpr.store import list_watchlist

    return {"entries": await list_watchlist(db, active_only=active_only, limit=limit)}


@router.post("/security/watchlist")
async def add_watchlist(payload: WatchlistIn, user=Depends(require_role("reviewer")),
                        db: AsyncSession = Depends(db_session)):
    """Watch a registration mark. Stored on its normalised form so however it is written, it is one entry."""
    from services.anpr.store import add_watchlist_entry

    try:
        return await add_watchlist_entry(
            db, plate=payload.plate, reason=payload.reason, severity=payload.severity,
            added_by=getattr(user, "name", None))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/security/watchlist/{entry_id}")
async def delete_watchlist(entry_id: str, db: AsyncSession = Depends(db_session)):
    """Deactivate an entry. Kept rather than deleted so an already-recorded hit stays explainable."""
    from services.anpr.store import remove_watchlist_entry

    return await remove_watchlist_entry(db, entry_id)


# ---------------------------------------------------------------- reads

@router.get("/security/reads")
async def get_reads(request: Request, user=Depends(require_role("reviewer")),
                    session_id: str | None = None, camera_id: str | None = None,
                    plate: str | None = None, state_code: str | None = None, hits_only: bool = False,
                    limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                    db: AsyncSession = Depends(db_session)):
    """The read feed, most recent first, filterable and paged.

    Reading this list is reading registration marks, which are personal data under DPDPA, so the access is
    recorded. This is the one place in the application where the plain text of somebody's plate is shown to
    a human on purpose, which makes it exactly the access an enquiry will ask about.
    """
    from services.anpr.store import list_reads
    from services.govern.pii_access import record_access

    out = await list_reads(db, session_id=session_id, camera_id=camera_id, plate=plate,
                           state_code=state_code, hits_only=hits_only, limit=limit, offset=offset)
    if out.get("reads"):
        await record_access(
            db, subject_type="plate_read", subject_id=(session_id or camera_id or "feed"),
            action="read_plate", user=user, session_id=session_id, pii_kinds=["plate"],
            # Unredacted, and recorded as such: the whole point of the security console is that the mark is
            # legible. Logging it as redacted would make the number that matters read as zero.
            redacted=False, route=str(request.url.path), pack_id="sec")
    return out


@router.get("/security/stats")
async def get_stats(db: AsyncSession = Depends(db_session)):
    """Headline counts for the console, including how many reads carry no measured confidence."""
    from services.anpr.store import read_stats

    return await read_stats(db)


class RecognizeIn(BaseModel):
    frame_id: str
    # Plate regions as [x1, y1, x2, y2, detector_score]. Supplied by the caller because plate localisation is
    # a detector's job, and the pack's detector seam is separate from the read path.
    regions: list[list[float]]
    camera_id: str | None = None


@router.post("/security/recognize")
async def recognize(payload: RecognizeIn, db: AsyncSession = Depends(db_session)):
    """Read the plates in the given regions of a frame, match them against the watchlist, and record them.

    The pack is resolved from the frame's own session, not from a request parameter: which domain a capture
    belongs to is a property of the capture, and letting a caller assert it would make the AV pack's refusal
    bypassable by anyone who could send a request.
    """
    import cv2
    import numpy as np

    from core.storage import get_object_store
    from db.models import Frame
    from services.anpr.recognize import recognize_plates
    from services.anpr.store import record_reads
    from services.domain import pack_id_for_session

    frame = await db.get(Frame, __import__("uuid").UUID(payload.frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")

    pack_id = await pack_id_for_session(db, frame.session_id)

    try:
        raw = get_object_store().get_bytes(frame.img_uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"could not read the frame image: {type(exc).__name__}") from exc
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "the frame image could not be decoded")

    # Built explicitly rather than via tuple(r): the recogniser takes a fixed 5-tuple, and a variadic tuple
    # would let a 4- or 6-element region through the length check into a signature that cannot hold it.
    regions = [(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
               for r in payload.regions if len(r) == 5]
    if not regions:
        raise HTTPException(400, "regions must be [x1, y1, x2, y2, score] quintuples")

    try:
        reads = recognize_plates(img, regions, pack_id=pack_id)
        return await record_reads(db, reads, session_id=str(frame.session_id),
                                  frame_id=payload.frame_id,
                                  camera_id=payload.camera_id or frame.cam_id, pack_id=pack_id)
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc
