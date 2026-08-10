"""Usage and invoices for metered deliveries.

Reads are reviewer-gated rather than open: an invoice reveals what a customer bought and when, which is
commercial information about someone other than the caller. Writing a certificate onto a delivery is an
admin action, since it upgrades an invoice line from "sold unmeasured" to "sold with a verifiable quality
claim" and that is a statement to a third party.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.billing import meter as meter_svc

router = APIRouter()


@router.get("/billing/usage", dependencies=[Depends(require_role("reviewer"))])
async def usage(account: str | None = None, since_ns: int | None = None, until_ns: int | None = None,
                db: AsyncSession = Depends(db_session)):
    """Usage totals by kind, with the certified and uncertified split inside each.

    The split is the part worth reading. "4.2 million objects delivered" and "0.3% of them carried a
    verifiable quality claim" are two halves of one sentence, and reporting only the first is how a data
    business ends up believing its own marketing.
    """
    return await meter_svc.usage_summary(db, account=account, since_ns=since_ns, until_ns=until_ns)


@router.get("/billing/invoice", dependencies=[Depends(require_role("reviewer"))])
async def invoice(account: str = "default", since_ns: int | None = None, until_ns: int | None = None,
                  db: AsyncSession = Depends(db_session)):
    """One line per delivery, so a disputed charge can be pointed at a specific sealed commit.

    `unpriced_lines` counts deliveries whose kind has no configured price. They are named rather than summed
    as zero, because a silent zero under-bills and looks exactly like a free tier.
    """
    return await meter_svc.invoice(db, account=account, since_ns=since_ns, until_ns=until_ns)


@router.post("/billing/certify/{commit_id}", dependencies=[Depends(require_role("admin"))])
async def certify(commit_id: str, eval_id: str, gold_id: str, model_version: str,
                  db: AsyncSession = Depends(db_session)):
    """Build the quality certificate for a shipped export and attach it to the invoice line.

    Exports normally ship before they are evaluated, so without this an export that was later measured would
    stay marked unmeasured forever. Builds the certificate here rather than accepting a signature from the
    caller: a client-supplied signature would let anybody mark a delivery certified without there being a
    certificate behind it, which is the whole thing this is meant to attest.
    """
    from core.config import get_settings
    from services.export.certificate import build_certificate

    key = get_settings().phase4.govern.attestation_key
    cert = await build_certificate(db, commit_id=commit_id, eval_id=eval_id, gold_id=gold_id,
                                   model_version=model_version, key=key)
    if "error" in cert:
        return cert
    record = await meter_svc.certify_delivery(db, kind="export", subject_id=commit_id,
                                              certificate_signature=cert["signature"])
    return {"certificate": cert["manifest"], "signature": cert["signature"], "usage_record": record}
