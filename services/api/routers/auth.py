"""Auth helpers for the web app. Currently just dev-login.

Deny-by-default auth (services/api/main.py) means a fresh browser with an empty localStorage cannot call the
gated read routes the shell loads on mount (/api/users, /api/datasets), so the app opens to a wall of 401s
with no way to sign in: the UserPicker itself reads /api/users. dev-login breaks that bootstrap deadlock by
handing the browser a real signed admin token.

It is hard-gated to `env == "local"`. That is the same boundary `_require_prod_secrets` uses to refuse the
weak default signing key, so a real deployment (env != local) both rejects this route AND would have a strong
key that makes a self-minted token impossible anyway. The `auth.dev_login` flag only lets you turn it off
locally to exercise the real login path; it cannot turn it on anywhere it is not already allowed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import User
from services.api.auth_token import mint_media_token, mint_token
from services.api.deps import db_session, require_user

router = APIRouter()
log = get_logger("auth")

# Fifteen minutes. Long enough that a review session does not stall re-minting between frames, short enough
# that a cookie captured from a proxy log is dead before anyone reads that log.
MEDIA_TOKEN_TTL_SECONDS = 900


@router.post("/auth/refresh")
async def refresh(user=Depends(require_user)) -> dict:
    """Mint a fresh token for the currently authenticated user so the web app can roll silently before the
    current one lapses. Requires a currently valid token: an expired or revoked token cannot refresh itself."""
    s = get_settings()
    return {"user_id": str(user.user_id), "role": user.role,
            "token": mint_token(user.user_id, s.auth.signing_key, token_version=user.token_version,
                                ttl_seconds=s.auth.token_ttl_seconds)}


@router.post("/auth/media-token")
async def media_token(user=Depends(require_user)) -> dict:
    """Issue the short-lived, image-only credential the browser puts in a cookie.

    Requires the full session token, so this is not a way to obtain access, only a way to downgrade access
    the caller already has into something safe to send automatically on every <img> request. The TTL is
    minutes: the cookie rides on requests whose URLs land in access logs and Referer headers, and a
    credential in that position should expire before it is worth stealing.
    """
    s = get_settings()
    ttl = MEDIA_TOKEN_TTL_SECONDS
    return {"token": mint_media_token(user.user_id, s.auth.signing_key,
                                      token_version=user.token_version, ttl_seconds=ttl),
            "expires_in": ttl}


@router.post("/auth/dev-login")
async def dev_login(db: AsyncSession = Depends(db_session)) -> dict:
    """Return a signed admin token for local development. 404 anywhere it is not permitted, so its existence
    is not even revealed off a dev box."""
    settings = get_settings()
    if settings.env != "local" or not settings.auth.dev_login:
        # 404, not 403: an endpoint that mints admin tokens should be invisible where it is not allowed.
        raise HTTPException(404, "not found")

    admin = (await db.execute(
        select(User).where(User.role == "admin").order_by(User.created_at).limit(1)
    )).scalars().first()
    if admin is None:
        # A brand-new local database has no users yet; seed the first admin so dev-login always resolves.
        admin = User(name="dev-admin", role="admin")
        db.add(admin)
        await db.commit()
        await db.refresh(admin)
        log.info("auth.dev_login_seeded_admin", user=str(admin.user_id))

    return {"user_id": str(admin.user_id), "name": admin.name, "role": admin.role,
            "token": mint_token(admin.user_id, settings.auth.signing_key,
                                token_version=admin.token_version,
                                ttl_seconds=settings.auth.token_ttl_seconds)}
