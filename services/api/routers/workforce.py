"""Routing labeling work to external teams, and receiving what they send back.

Two different callers reach this router and they authenticate differently. An operator registering a vendor
or dispatching a job carries a bearer token and a role, as everywhere else. A workforce returning a batch
does not: it is an outside party with no user account, so its callback is authenticated by an HMAC signature
over the body using the secret minted at registration, verified with the same helper webhooks use, including
the replay window. Letting a vendor callback through the normal auth would mean issuing vendors user
accounts, and a vendor with a user account can do considerably more than return a batch.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.labelops import workforce as wf_svc

router = APIRouter()


class WorkforceIn(BaseModel):
    name: str
    kind: str = "vendor"
    endpoint: str | None = None
    capabilities: dict = {}
    capacity_jobs_per_day: int = 0
    min_honeypot_accuracy: float = 0.9
    contact: str | None = None


@router.post("/workforce", dependencies=[Depends(require_role("admin"))])
async def register(payload: WorkforceIn, db: AsyncSession = Depends(db_session)):
    """Register a workforce and mint its callback secret.

    Admin, because registering a workforce creates a credential that can write annotations into the corpus.
    The secret is returned exactly once, here; no read path returns it again.
    """
    try:
        return await wf_svc.register_workforce(
            db, name=payload.name, kind=payload.kind, endpoint=payload.endpoint,
            capabilities=payload.capabilities, capacity_jobs_per_day=payload.capacity_jobs_per_day,
            min_honeypot_accuracy=payload.min_honeypot_accuracy, contact=payload.contact)
    except wf_svc.WorkforceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class DispatchIn(BaseModel):
    job_id: str
    workforce_id: str | None = None      # omitted means "route it"
    required_classes: list[str] | None = None
    deliver: bool = True


@router.post("/workforce/dispatch", dependencies=[Depends(require_role("reviewer"))])
async def dispatch(payload: DispatchIn, db: AsyncSession = Depends(db_session)):
    """Send a job to a workforce, choosing one automatically when none is named.

    Reviewer, matching assign_job: directing other people's time is a reviewer action.
    """
    try:
        workforce_id = payload.workforce_id
        routed = None
        if not workforce_id:
            routed = await wf_svc.route_job(db, job_id=payload.job_id,
                                            required_classes=payload.required_classes)
            workforce_id = routed.get("workforce_id")
            if not workforce_id:
                return {"dispatched": False, **routed}
        result = await wf_svc.dispatch_job(db, job_id=payload.job_id, workforce_id=workforce_id,
                                           deliver=payload.deliver)
        return {"dispatched": True, "routed": routed, **result}
    except wf_svc.WorkforceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/workforce/rating", dependencies=[Depends(require_role("reviewer"))])
async def rating(db: AsyncSession = Depends(db_session)):
    """Accept rate per workforce, with the interval and the count.

    `routing_weight` is the lower bound, not the point estimate: three accepted batches out of three is 1.0
    and means very little. A workforce below the evidence floor is marked unproven rather than poor.
    """
    return await wf_svc.workforce_rating(db)


@router.get("/workforce/{workforce_id}/pending")
async def pending(workforce_id: str, limit: int = 50,
                  x_labelox_signature: str = Header(default=""),
                  db: AsyncSession = Depends(db_session)):
    """Open dispatches, for a workforce that polls rather than hosting a callback endpoint.

    Signature-authenticated like the return path, because the caller is the vendor rather than an operator.
    Requiring every vendor to host an HTTPS callback would exclude exactly the smaller outfits an
    India-focused operation would want to work with.
    """
    wf = await _authenticate(db, workforce_id, b"", x_labelox_signature)
    return {"workforce": wf.name, "pending": await wf_svc.pending_for_workforce(db, workforce_id, limit)}


class ReturnIn(BaseModel):
    assignment_id: str
    external_ref: str | None = None
    objects_returned: int = 0
    detail: dict = {}


@router.post("/workforce/{workforce_id}/return")
async def submit_return(workforce_id: str, request: Request,
                        x_labelox_signature: str = Header(default=""),
                        db: AsyncSession = Depends(db_session)):
    """A workforce returns a finished batch. Honeypots decide whether it is accepted.

    The raw body is read and verified before it is parsed, because the signature covers the bytes that were
    sent. Parsing first and re-serialising would verify a reconstruction of the message rather than the
    message, and key ordering or spacing would break it in ways that look like an attack.
    """
    raw = await request.body()
    wf = await _authenticate(db, workforce_id, raw, x_labelox_signature)

    try:
        payload = ReturnIn.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"unparseable return body: {exc}") from exc

    asg = await wf_svc.get_assignment(db, payload.assignment_id)
    if asg is None:
        raise HTTPException(status_code=404, detail="assignment not found")
    if str(asg.workforce_id) != str(wf.workforce_id):
        # A valid signature from workforce A must not settle workforce B's assignment.
        raise HTTPException(status_code=403, detail="assignment belongs to a different workforce")

    try:
        return await wf_svc.submit_return(db, assignment_id=payload.assignment_id,
                                          external_ref=payload.external_ref,
                                          objects_returned=payload.objects_returned,
                                          detail=payload.detail)
    except wf_svc.WorkforceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _authenticate(db: AsyncSession, workforce_id: str, body: bytes, signature: str):
    """Resolve a workforce from its id and verify the request signature over the exact bytes sent."""
    wf = await wf_svc.get_workforce(db, workforce_id)
    if wf is None or not wf.active:
        # Same answer for unknown and inactive, so this cannot be used to enumerate registered vendors.
        raise HTTPException(status_code=403, detail="unknown or inactive workforce")
    if not wf_svc.verify_return_signature(wf.secret, body, signature):
        raise HTTPException(status_code=403, detail="bad signature")
    return wf
