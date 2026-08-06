"""LabeloxSec v2: zones, incidents, cross-camera identity, and live ingest.

Every route here is capability-gated in the service beneath it, not merely role-gated. Drawing a tripwire
on a camera and following a person between cameras are lawful for an authorised security deployment and are
exactly what the AV pack must never do, so the refusal lives in the pack rather than in a role check that
an admin token would satisfy.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.anpr.recognize import AnprNotAuthorised
from services.api.deps import db_session, require_role, require_user

# Reviewer floor throughout: incidents and identities are personal data and operational security state.
router = APIRouter(dependencies=[Depends(require_role("reviewer"))])


def _refuse(exc: AnprNotAuthorised) -> HTTPException:
    return HTTPException(status_code=403, detail=str(exc))


# ---------------------------------------------------------------- zones

class ZoneIn(BaseModel):
    camera_id: str
    name: str
    points: list[list[float]]
    kind: str = "area"          # area | line
    rule: str = "enter"         # enter | exit | dwell | cross
    classes: list[str] = []
    dwell_seconds: float | None = None
    severity: str = "warn"
    session_id: str | None = None
    # Which domain this zone belongs to. Stated rather than inherited from the deployment default, because
    # a deployment can run both packs and the default is the AV one: a zone request that silently adopted
    # it would be refused on a deployment where the sec pack is present and perfectly usable. When a
    # session is given, the session's own pack wins, since which domain a capture belongs to is a property
    # of the capture rather than of the request.
    pack_id: str = "sec"


@router.get("/sec/zones")
async def list_zones(camera_id: str | None = None, active_only: bool = True,
                     db: AsyncSession = Depends(db_session)):
    from services.sec.incidents import list_zones as _list

    return await _list(db, camera_id=camera_id, active_only=active_only)


@router.post("/sec/zones")
async def create_zone(payload: ZoneIn, user=Depends(require_user),
                      db: AsyncSession = Depends(db_session)):
    from services.domain import pack_id_for_session
    from services.sec.incidents import IncidentError
    from services.sec.incidents import create_zone as _create

    body = payload.model_dump()
    if body.get("session_id"):
        body["pack_id"] = await pack_id_for_session(db, body["session_id"])
    try:
        return await _create(db, **body, created_by=getattr(user, "name", None))
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc
    except IncidentError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/sec/zones/{zone_id}")
async def delete_zone(zone_id: str, db: AsyncSession = Depends(db_session)):
    from services.sec.incidents import delete_zone as _delete

    return await _delete(db, zone_id)


@router.post("/sec/sessions/{session_id}/evaluate")
async def evaluate_session(session_id: str, db: AsyncSession = Depends(db_session)):
    """Run every active zone for this session's camera over its tracks, and raise what fires."""
    from services.sec.incidents import IncidentError
    from services.sec.incidents import evaluate_session as _eval

    try:
        return await _eval(db, session_id)
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc
    except IncidentError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- incidents

@router.get("/sec/incidents")
async def list_incidents(camera_id: str | None = None, kind: str | None = None,
                         status: str | None = None, severity: str | None = None,
                         since_hours: int | None = None,
                         limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                         db: AsyncSession = Depends(db_session)):
    from services.sec.incidents import list_incidents as _list

    return await _list(db, camera_id=camera_id, kind=kind, status=status, severity=severity,
                       since_hours=since_hours, limit=limit, offset=offset)


@router.post("/sec/incidents/{incident_id}/acknowledge")
async def acknowledge(incident_id: str, close: bool = False, user=Depends(require_user),
                      db: AsyncSession = Depends(db_session)):
    from services.sec.incidents import IncidentError
    from services.sec.incidents import acknowledge as _ack

    try:
        return await _ack(db, incident_id, by=getattr(user, "name", None), close=close)
    except IncidentError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/sec/sessions/{session_id}/stitch-plates")
async def stitch_plates(session_id: str, db: AsyncSession = Depends(db_session)):
    """Attach plate reads to the incidents that happened at the same moment on the same camera.

    The join an operator was doing in their head: a crossing at 14:02:11 and a plate read at 14:02:11 on
    the same camera are one van.
    """
    from services.domain import pack_id_for_session
    from services.sec.incidents import stitch_plate_reads

    try:
        return await stitch_plate_reads(db, session_id,
                                        pack_id=await pack_id_for_session(db, session_id))
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc


# ---------------------------------------------------------------- re-identification

@router.get("/sec/identities")
async def list_identities(min_cameras: int = Query(1, ge=1, le=20),
                          limit: int = Query(100, ge=1, le=500),
                          db: AsyncSession = Depends(db_session)):
    """Appearance signatures, never names. The signature itself is never returned."""
    from services.sec.reid import list_identities as _list

    return await _list(db, min_cameras=min_cameras, limit=limit)


@router.get("/sec/identities/{identity_id}")
async def identity_detail(identity_id: str, db: AsyncSession = Depends(db_session)):
    from services.sec.reid import ReidError
    from services.sec.reid import identity_detail as _detail

    try:
        return await _detail(db, identity_id)
    except ReidError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/sec/identities/{identity_id}")
async def forget_identity(identity_id: str, db: AsyncSession = Depends(db_session)):
    """Erase a signature and everything attributed to it.

    Present because it must be: a signature is derived from a person's appearance, so it is personal data
    under DPDPA whether or not a name is attached, and an erasure request has to be able to reach it.
    """
    from services.sec.reid import forget_identity as _forget

    return await _forget(db, identity_id)


@router.post("/sec/sessions/{session_id}/link-identities")
async def link_identities(session_id: str, db: AsyncSession = Depends(db_session)):
    from services.domain import pack_id_for_session
    from services.sec.reid import link_session

    try:
        # Resolved from the capture, not asserted by the caller: letting a request name its own domain
        # would make the AV pack's refusal bypassable by anyone who could send one.
        return await link_session(db, session_id,
                                  pack_id=await pack_id_for_session(db, session_id))
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc


# ---------------------------------------------------------------- live ingest

class RtspIn(BaseModel):
    url: str
    camera_id: str
    city: str | None = None
    pack_id: str = "sec"
    max_frames: int = 500
    max_seconds: float = 300.0
    motion_threshold: float = 6.0
    heartbeat_seconds: float = 30.0


@router.post("/sec/rtsp/ingest")
async def rtsp_ingest(payload: RtspIn, db: AsyncSession = Depends(db_session)):  # noqa: ARG001
    """Sample a live camera into a session.

    Bounded by frames and by seconds, both. A live stream has no end, so a request without a limit is a
    request to run forever, and the caller almost never means that.
    """
    from services.domain import default_pack_id, get_pack

    pid = payload.pack_id or default_pack_id()
    source = getattr(get_pack(pid), "stream_source", None)
    if source is None:
        raise HTTPException(400, f"pack {pid!r} has no live-stream source; it cannot sample an RTSP camera")

    policy = source.sampling_policy(motion_threshold=payload.motion_threshold,
                                    heartbeat_seconds=payload.heartbeat_seconds)
    try:
        return await source.ingest(payload.url, payload.camera_id, city=payload.city,
                                   policy=policy, max_frames=payload.max_frames,
                                   max_seconds=payload.max_seconds, pack_id=payload.pack_id)
    except AnprNotAuthorised as exc:
        raise _refuse(exc) from exc
    except source.unavailable_error as exc:
        # 502, not 500: the camera is an upstream this server could not reach, which is a different
        # problem from this server being broken.
        raise HTTPException(502, str(exc)) from exc
