"""Sign-in, self-service, and federated identity: the routes a person uses to obtain and keep access.

Before this the only credential path was an admin running a script, so there was no sign-in page worth the
name, no way to recover an account, and no second factor. Everything here exists to close that.

Two rules run through the whole file:

- **A failed sign-in never says which half was wrong.** "No such user" and "wrong password" as separate
  answers turn the login form into an account-enumeration oracle, and the enumerated list is exactly what a
  credential-stuffing run needs. Timing is levelled the same way: an unknown user still pays for a hash.
- **Nothing here is reachable with a partially-authenticated token.** MFA is completed before any token is
  issued, using a short-lived challenge that can do nothing except finish that one sign-in. Issuing a real
  token and then asking for the code would mean the first factor alone already granted access.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import User, UserCredential
from services.api import identity as ident
from services.api.auth_token import mint_token
from services.api.deps import db_session, require_user

router = APIRouter()
log = get_logger("identity_api")

# A dummy hash every unknown-user attempt is verified against. Without it, "no such user" returns in
# microseconds while a real user pays for scrypt, and the difference alone enumerates accounts.
_DUMMY_HASH = ident.hash_password("a-password-that-is-never-anyone-s-actual-password")

# Half-finished sign-ins: first factor accepted, second factor outstanding. Held server-side with a short
# life so the handle can do nothing but finish this one login.
_MFA_TTL_SECONDS = 300
_mfa_pending: dict[str, tuple[str, float]] = {}


def _mfa_begin(user_id: uuid.UUID) -> str:
    now = datetime.now(UTC).timestamp()
    for k, (_, at) in list(_mfa_pending.items()):
        if now - at > _MFA_TTL_SECONDS:
            _mfa_pending.pop(k, None)
    handle = secrets.token_urlsafe(24)
    _mfa_pending[handle] = (str(user_id), now)
    return handle


def _mfa_take(handle: str) -> str:
    entry = _mfa_pending.pop(handle or "", None)   # one handle, one attempt
    if entry is None:
        raise HTTPException(400, "this sign-in has expired; start again")
    uid, at = entry
    if datetime.now(UTC).timestamp() - at > _MFA_TTL_SECONDS:
        raise HTTPException(400, "this sign-in has expired; start again")
    return uid


def _mfa_required(role: str) -> bool:
    return role in set(get_settings().auth.mfa_required_roles or [])


async def _issue(db: AsyncSession, user: User) -> dict:
    s = get_settings()
    from services.activity import record_activity

    await record_activity(db, user=user, verb="signed_in", summary=f"{user.name} signed in")
    return {"user_id": str(user.user_id), "name": user.name, "role": user.role,
            "token": mint_token(user.user_id, s.auth.signing_key, token_version=user.token_version,
                                ttl_seconds=s.auth.token_ttl_seconds)}


# ---------------------------------------------------------------- what the login page renders

@router.get("/auth/methods")
async def auth_methods(db: AsyncSession = Depends(db_session)) -> dict:
    """Which sign-in methods this deployment offers.

    Public by design and deliberately contentless: it reveals configuration, never whether any particular
    account exists. The login page needs it to decide what to draw rather than showing a password form on a
    directory-only deployment.
    """
    s = get_settings().auth
    n_users = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    return {
        "password": bool(s.password_login),
        "self_signup": bool(s.self_signup),
        "signup_domains": list(s.self_signup_domains or []),
        "oidc": bool(s.oidc_enabled),
        "oidc_issuer": s.oidc_issuer,
        "dev_login": bool(get_settings().env == "local" and s.dev_login),
        # The genuine first-run case: no accounts exist, so the first person to arrive may create the admin.
        "bootstrap": int(n_users) == 0,
    }


# ---------------------------------------------------------------- password sign-in

class LoginIn(BaseModel):
    name: str
    password: str


@router.post("/auth/login")
async def login(payload: LoginIn, db: AsyncSession = Depends(db_session)) -> dict:
    s = get_settings().auth
    if not s.password_login:
        raise HTTPException(400, "password sign-in is disabled on this deployment")

    user = (await db.execute(
        select(User).where(User.name == payload.name.strip()))).scalar_one_or_none()
    cred = await db.get(UserCredential, user.user_id) if user else None

    # Locked accounts are told so plainly. Hiding it would leave a user retrying against a wall with no
    # idea why, and the lockout state is not itself a secret worth protecting.
    remaining = ident.lockout_remaining(cred)
    if remaining:
        raise HTTPException(429, f"too many attempts; try again in {remaining // 60 + 1} minutes")

    # The unknown-user branch still pays for a hash, so the two paths cost the same.
    ok = ident.verify_password(payload.password, cred.password_hash if cred else _DUMMY_HASH)
    if user is None or cred is None or not cred.password_hash or not ok:
        if cred is not None:
            await ident.record_failure(db, cred)
        log.warning("identity.login_failed", name=payload.name[:32])
        raise HTTPException(401, "that user name and password do not match")

    await ident.record_success(db, cred)

    if cred.totp_confirmed_at:
        return {"mfa_required": True, "mfa_handle": _mfa_begin(user.user_id),
                "methods": ["totp", "recovery"] if cred.recovery_hashes else ["totp"]}
    if _mfa_required(user.role):
        # Enrolment is compelled at the door rather than granting a token first: a role that requires a
        # second factor must not be usable before one exists.
        return {"mfa_required": True, "mfa_enrol": True, "mfa_handle": _mfa_begin(user.user_id),
                "detail": f"the {user.role} role requires a second factor; set one up to continue"}
    return await _issue(db, user)


class MfaIn(BaseModel):
    mfa_handle: str
    code: str


@router.post("/auth/login/mfa")
async def login_mfa(payload: MfaIn, db: AsyncSession = Depends(db_session)) -> dict:
    """Finish a sign-in with the second factor. Accepts a TOTP code or a single-use recovery code."""
    uid = _mfa_take(payload.mfa_handle)
    user = await db.get(User, uuid.UUID(uid))
    cred = await db.get(UserCredential, uuid.UUID(uid))
    if user is None or cred is None or not cred.totp_secret:
        raise HTTPException(400, "this sign-in has expired; start again")

    step = ident.verify_totp(cred.totp_secret, payload.code, last_step=cred.last_totp_step)
    if step is not None:
        cred.last_totp_step = step        # blocks replay inside the code's own window
        await db.commit()
        return await _issue(db, user)

    if ident.consume_recovery(cred, payload.code):
        await db.commit()
        log.warning("identity.recovery_code_used", user=uid,
                    remaining=len(cred.recovery_hashes or []))
        return await _issue(db, user)

    await ident.record_failure(db, cred)
    raise HTTPException(401, "that code is not valid")


# ---------------------------------------------------------------- self-service

class SignupIn(BaseModel):
    name: str
    password: str
    email: str | None = None


@router.post("/auth/signup")
async def signup(payload: SignupIn, db: AsyncSession = Depends(db_session)) -> dict:
    """Create an account.

    Two quite different situations share this route. On a deployment with self-signup enabled it registers
    an ordinary account at the configured role. On a brand-new deployment with no users at all it creates
    the first administrator, which is the bootstrap that otherwise required shell access to the server.
    """
    s = get_settings().auth
    n_users = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    bootstrap = int(n_users) == 0
    if not bootstrap and not s.self_signup:
        raise HTTPException(403, "self-registration is disabled; ask an administrator for an account")

    name = payload.name.strip()
    if not name or len(name) > 64 or not all(c.isalnum() or c in "-._" for c in name):
        raise HTTPException(400, "user name must be alphanumeric with - . or _ and at most 64 characters")

    domains = [d.lower().lstrip("@") for d in (s.self_signup_domains or [])]
    if domains and not bootstrap:
        addr = (payload.email or "").strip().lower()
        if "@" not in addr or addr.rsplit("@", 1)[-1] not in domains:
            raise HTTPException(403, f"registration is limited to {', '.join(domains)} addresses")

    if (await db.execute(select(User).where(User.name == name))).scalar_one_or_none() is not None:
        raise HTTPException(409, "that user name is taken")

    try:
        ident.check_password_policy(payload.password, name=name)
    except ident.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc

    user = User(name=name, role="admin" if bootstrap else (s.self_signup_role or "annotator"))
    db.add(user)
    await db.flush()
    cred = await ident.get_or_create_credential(db, user.user_id)
    cred.email = (payload.email or "").strip() or None
    await ident.set_password(db, user, payload.password)   # commits
    log.info("identity.signup", user=str(user.user_id), bootstrap=bootstrap, role=user.role)

    if not bootstrap and _mfa_required(user.role):
        return {"mfa_required": True, "mfa_enrol": True, "mfa_handle": _mfa_begin(user.user_id)}
    return await _issue(db, user)


class ResetRequestIn(BaseModel):
    name: str


@router.post("/auth/password/reset-request")
async def reset_request(payload: ResetRequestIn, db: AsyncSession = Depends(db_session)) -> dict:
    """Begin a password reset.

    Always answers the same way, whether or not the account exists, for the enumeration reason above. There
    is no mail path here on purpose: a deployment's delivery is its own, and either a hard SMTP dependency
    or a silent no-op that looks like it worked would be worse than saying plainly that an administrator
    hands the link over. On a local deployment the link is returned directly so the flow is usable.
    """
    user = (await db.execute(
        select(User).where(User.name == payload.name.strip()))).scalar_one_or_none()
    out: dict = {"requested": True,
                 "detail": "if that account exists, a reset link has been created for it"}
    if user is None:
        return out
    token = await ident.create_reset(db, user)
    if get_settings().env == "local":
        out["reset_token"] = token
        out["detail"] = "local deployment: use this token directly"
    else:
        # Recorded so an administrator can hand it over out of band. Never logged in full.
        log.info("identity.reset_pending", user=str(user.user_id), token_prefix=token[:6])
    return out


class ResetIn(BaseModel):
    token: str
    password: str


@router.post("/auth/password/reset")
async def reset(payload: ResetIn, db: AsyncSession = Depends(db_session)) -> dict:
    try:
        user = await ident.consume_reset(db, payload.token, payload.password)
    except ident.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await _issue(db, user)


class ChangePasswordIn(BaseModel):
    current_password: str | None = None
    new_password: str


@router.post("/auth/password/change")
async def change_password(payload: ChangePasswordIn, user=Depends(require_user),
                          db: AsyncSession = Depends(db_session)) -> dict:
    """Change your own password. Proving the current one is required whenever one is set, so a borrowed
    session cannot be turned into permanent ownership of the account."""
    cred = await ident.get_or_create_credential(db, user.user_id)
    if cred.password_hash and not ident.verify_password(payload.current_password or "",
                                                        cred.password_hash):
        raise HTTPException(401, "your current password is not correct")
    try:
        await ident.set_password(db, user, payload.new_password)
    except ident.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc
    # set_password bumped token_version, so the caller's own token is now stale too. Hand back a fresh one
    # rather than signing them out of the tab they are standing in.
    return await _issue(db, user)


@router.get("/auth/profile")
async def profile(user=Depends(require_user), db: AsyncSession = Depends(db_session)) -> dict:
    cred = await db.get(UserCredential, user.user_id)
    return ident.public_state(user, cred)


# ---------------------------------------------------------------- second factor

@router.post("/auth/mfa/setup")
async def mfa_setup(user=Depends(require_user), db: AsyncSession = Depends(db_session)) -> dict:
    """Generate a secret and return the otpauth URI to scan. Not active until confirmed."""
    cred = await ident.get_or_create_credential(db, user.user_id)
    if cred.totp_confirmed_at:
        raise HTTPException(409, "a second factor is already enrolled; remove it first")
    cred.totp_secret = ident.new_totp_secret()
    await db.commit()
    return {"secret": cred.totp_secret,
            "otpauth_uri": ident.totp_uri(cred.totp_secret, user.name),
            "detail": "scan this, then confirm with a code to activate it"}


class PendingSetupIn(BaseModel):
    mfa_handle: str


@router.post("/auth/mfa/setup-pending")
async def mfa_setup_pending(payload: PendingSetupIn, db: AsyncSession = Depends(db_session)) -> dict:
    """Generate a secret for a sign-in that was stopped because the role requires a second factor.

    Separate from /auth/mfa/setup because at this point the caller holds no token: the whole reason they are
    here is that the first factor alone must not grant one. The challenge handle is the authority, it is
    re-issued so the flow can continue, and it can still do nothing but finish this one sign-in.
    """
    uid = _mfa_take(payload.mfa_handle)
    user = await db.get(User, uuid.UUID(uid))
    if user is None:
        raise HTTPException(400, "this sign-in has expired; start again")
    cred = await ident.get_or_create_credential(db, user.user_id)
    if cred.totp_confirmed_at:
        raise HTTPException(409, "a second factor is already enrolled")
    cred.totp_secret = ident.new_totp_secret()
    await db.commit()
    return {"secret": cred.totp_secret, "otpauth_uri": ident.totp_uri(cred.totp_secret, user.name),
            "mfa_handle": _mfa_begin(user.user_id)}


class CodeIn(BaseModel):
    code: str
    mfa_handle: str | None = None


@router.post("/auth/mfa/confirm")
async def mfa_confirm(payload: CodeIn, db: AsyncSession = Depends(db_session),
                      request: Request = None) -> dict:  # noqa: ARG001
    """Activate an enrolled factor and hand back the recovery codes, which are shown exactly once.

    Reachable two ways: by an already-signed-in user hardening their account, or during a sign-in that was
    stopped because the role requires a factor and none existed. The second path carries the handle from
    that sign-in, so enrolment finishes the login it interrupted.
    """
    if payload.mfa_handle:
        uid = _mfa_take(payload.mfa_handle)
        user = await db.get(User, uuid.UUID(uid))
        if user is None:
            raise HTTPException(400, "this sign-in has expired; start again")
        cred = await db.get(UserCredential, user.user_id)
    else:
        user = await require_user(await _current_from_request(request, db))
        cred = await db.get(UserCredential, user.user_id)

    if cred is None or not cred.totp_secret:
        raise HTTPException(400, "start the enrolment first")
    step = ident.verify_totp(cred.totp_secret, payload.code, last_step=cred.last_totp_step)
    if step is None:
        raise HTTPException(401, "that code is not valid")

    codes = ident.new_recovery_codes()
    cred.recovery_hashes = [ident.hash_recovery(c) for c in codes]
    cred.totp_confirmed_at = datetime.now(UTC)
    cred.last_totp_step = step
    await db.commit()
    log.info("identity.mfa_enrolled", user=str(user.user_id))
    return {**(await _issue(db, user)), "recovery_codes": codes,
            "detail": "save these recovery codes now; they are not shown again"}


class DisableMfaIn(BaseModel):
    password: str


@router.post("/auth/mfa/disable")
async def mfa_disable(payload: DisableMfaIn, user=Depends(require_user),
                      db: AsyncSession = Depends(db_session)) -> dict:
    """Remove the second factor. Requires the password, because a borrowed session must not be able to
    strip the protection that exists to contain exactly that."""
    cred = await db.get(UserCredential, user.user_id)
    if cred is None or not cred.totp_confirmed_at:
        raise HTTPException(404, "no second factor is enrolled")
    if not ident.verify_password(payload.password, cred.password_hash):
        raise HTTPException(401, "your password is not correct")
    if _mfa_required(user.role):
        raise HTTPException(403, f"the {user.role} role requires a second factor")
    cred.totp_secret = None
    cred.totp_confirmed_at = None
    cred.recovery_hashes = []
    cred.last_totp_step = None
    await db.commit()
    log.warning("identity.mfa_disabled", user=str(user.user_id))
    return {"mfa_enabled": False}


@router.post("/auth/sessions/revoke")
async def revoke_sessions(user=Depends(require_user), db: AsyncSession = Depends(db_session)) -> dict:
    """Sign out everywhere. Bumps the token version, which invalidates every token already issued to this
    user at once, and returns a fresh one so the caller keeps the tab they are in."""
    user.token_version = int(user.token_version) + 1
    await db.commit()
    log.info("identity.sessions_revoked", user=str(user.user_id))
    return await _issue(db, user)


# ---------------------------------------------------------------- OIDC

_OIDC_COOKIE = "lbx_oidc"


@router.get("/auth/oidc/start")
async def oidc_start(next: str = "/", db: AsyncSession = Depends(db_session)):  # noqa: A002, ARG001
    """Redirect the browser to the identity provider."""
    from starlette.responses import RedirectResponse

    from services.api import oidc

    s = get_settings().auth
    if not s.oidc_enabled:
        raise HTTPException(404, "single sign-on is not configured on this deployment")
    try:
        doc = await oidc.discover(s.oidc_issuer)
        url, handle = oidc.begin(doc, client_id=s.oidc_client_id, redirect_uri=s.oidc_redirect_uri,
                                 scopes=s.oidc_scopes, next_url=_safe_next(next))
    except oidc.OidcError as exc:
        raise HTTPException(502, str(exc)) from exc

    resp = RedirectResponse(url, status_code=302)
    # HttpOnly: the handle is the only thing tying the callback to this browser, so script must not read it.
    # Lax rather than Strict because the provider's redirect is a cross-site navigation and Strict would
    # withhold the cookie exactly when it is needed.
    resp.set_cookie(_OIDC_COOKIE, handle, max_age=600, httponly=True, samesite="lax", path="/api")
    return resp


@router.get("/auth/oidc/callback")
async def oidc_callback(request: Request, code: str | None = None, state: str | None = None,
                        error: str | None = None, db: AsyncSession = Depends(db_session)):
    """Complete the exchange and hand the app a token."""
    from starlette.responses import RedirectResponse

    from services.api import oidc

    s = get_settings().auth
    if not s.oidc_enabled:
        raise HTTPException(404, "single sign-on is not configured on this deployment")
    if error:
        raise HTTPException(401, f"the identity provider refused the sign-in: {error}")
    if not code:
        raise HTTPException(400, "the identity provider returned no authorization code")

    try:
        pending = oidc.take_pending(request.cookies.get(_OIDC_COOKIE))
        if not state or not secrets.compare_digest(state, pending.state):
            # Without this a code from another session could be injected into this browser's login.
            raise oidc.OidcError("this sign-in did not originate here")
        doc = await oidc.discover(s.oidc_issuer)
        tokens = await oidc.exchange(doc, code=code, pending=pending, client_id=s.oidc_client_id,
                                     client_secret=s.oidc_client_secret)
        id_token = tokens.get("id_token")
        if not id_token:
            raise oidc.OidcError("the identity provider returned no identity token")
        jwks = await oidc.fetch_jwks(doc["jwks_uri"])
        claims = oidc.verify_id_token(id_token, jwks=jwks, issuer=s.oidc_issuer,
                                      audience=s.oidc_client_id, nonce=pending.nonce)
    except oidc.OidcError as exc:
        raise HTTPException(401, str(exc)) from exc

    user, created = await ident.link_or_create_oidc_user(
        db, issuer=s.oidc_issuer.rstrip("/"), subject=str(claims["sub"]),
        email=claims.get("email"), preferred_name=claims.get("preferred_username") or claims.get("name"),
        default_role=s.oidc_default_role)
    issued = await _issue(db, user)
    log.info("identity.oidc_login", user=str(user.user_id), created=created)

    # Hand the token to the app through a one-shot, readable cookie the login page immediately consumes and
    # clears. A token in the URL fragment would survive in history; in the query string it would reach the
    # server log of every proxy on the way.
    resp = RedirectResponse(pending.next_url or "/", status_code=302)
    resp.delete_cookie(_OIDC_COOKIE, path="/api")
    resp.set_cookie("lbx_sso_handoff", ident.json_safe(issued), max_age=60,
                    httponly=False, samesite="lax", path="/")
    return resp


def _safe_next(nxt: str) -> str:
    """Only a same-site path. An absolute URL here is an open redirect, which is how a phishing page gets
    to wear this application's sign-in flow."""
    n = (nxt or "/").strip()
    return n if n.startswith("/") and not n.startswith("//") else "/"


async def _current_from_request(request: Request, db: AsyncSession):
    from services.api.deps import current_user

    return await current_user(request=request, authorization=request.headers.get("authorization"),
                              x_lbx_user_id=None, db=db)


@router.post("/auth/logout")
async def logout(response: Response) -> dict:
    """Clear the cookies this app sets. The token itself is stateless, so the client drops it; revoking
    every session is a separate, deliberate action."""
    response.delete_cookie("lbx_media", path="/api")
    response.delete_cookie("lbx_sso_handoff", path="/")
    return {"signed_out": True}
