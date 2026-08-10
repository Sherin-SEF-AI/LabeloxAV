"""Managing machine credentials.

Admin-only throughout, and the router carries no read exemption: the listing shows which integrations can
write to the corpus and under what role, which is not an open read.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import current_user, db_session, require_role
from services.identity.service_accounts import (
    ServiceAccountError,
    listing,
    mint,
    revoke,
    rotate,
)

router = APIRouter(dependencies=[Depends(require_role("admin"))])


class MintIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    role: str = "annotator"
    description: str | None = None
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    scopes: list[str] | None = None


@router.post("/service-accounts")
async def create(body: MintIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Mint a machine credential. The key is in this response and nowhere else, ever."""
    try:
        return await mint(db, name=body.name, role=body.role, description=body.description,
                          created_by=user.user_id if user else None,
                          expires_in_days=body.expires_in_days, scopes=body.scopes)
    except ServiceAccountError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/service-accounts")
async def index(include_revoked: bool = False, db: AsyncSession = Depends(db_session)):
    """Every account, without the keys, which are not recoverable."""
    return {"accounts": await listing(db, include_revoked=include_revoked)}


@router.post("/service-accounts/{account_id}/revoke")
async def kill(account_id: str, db: AsyncSession = Depends(db_session)):
    """Stop this key working now. Idempotent."""
    out = await revoke(db, _uuid(account_id))
    if out is None:
        raise HTTPException(404, "service account not found")
    return out


@router.post("/service-accounts/{account_id}/rotate")
async def new_key(account_id: str, db: AsyncSession = Depends(db_session)):
    """Replace the secret, keeping the identity, so a leaked key is swapped without re-pointing whatever it
    authorises at a different account."""
    try:
        out = await rotate(db, _uuid(account_id))
    except ServiceAccountError as exc:
        raise HTTPException(409, str(exc)) from exc
    if out is None:
        raise HTTPException(404, "service account not found")
    return out


def _uuid(s: str) -> uuid.UUID:
    try:
        return uuid.UUID(s)
    except ValueError as exc:
        raise HTTPException(400, "invalid service account id") from exc
