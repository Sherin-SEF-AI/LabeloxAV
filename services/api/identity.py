"""Passwords, second factors, and federated identity.

Until this existed the only credential was an admin-minted token: a person could not obtain access
themselves, could not add a second factor, and could not sign in through a corporate directory. That was the
largest single blocker to a real deployment, and it was a product decision rather than a coding task. The
decision taken here is the conservative one: local passwords with TOTP, plus OIDC for deployments that
already have a directory, and no third-party dependency for either.

Choices worth stating, because each is a trade rather than an obvious default:

- **scrypt, from hashlib.** No argon2 or bcrypt dependency. scrypt is memory-hard, ships in the standard
  library, and its parameters are encoded in the stored string, so the cost can be raised later per user on
  next sign-in instead of forcing a global reset.
- **Lockout in the database, not a cache.** A lockout that a process restart clears is not a lockout, and
  this deployment can run more than one API replica.
- **TOTP replay is blocked by recording the accepted step.** A code is valid for a whole window, so without
  this an attacker who observes one has the rest of that window to reuse it.
- **A half-enrolled second factor does not gate sign-in.** Requiring an unconfirmed secret would lock a user
  out of their own account with an authenticator they never finished setting up.
- **OIDC identity is issuer plus subject, never email.** A directory can reassign an address to a different
  person, and matching on it would hand them the previous holder's account.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import PasswordReset, User, UserCredential

log = get_logger("identity")

# scrypt work factors. n=2**15 with r=8 costs about 32 MB and a few tens of milliseconds, which is the usual
# balance between resisting an offline attack and not turning a sign-in into a denial of service.
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

MIN_PASSWORD_LEN = 12
MAX_FAILED_ATTEMPTS = 8
LOCKOUT_MINUTES = 15
RESET_TTL_MINUTES = 30
RECOVERY_CODES = 10

# A short list of the passwords an attacker tries first. Length alone does not save "passwordpassword".
_OBVIOUS = {
    "password", "passw0rd", "letmein", "welcome", "iloveyou", "admin", "administrator",
    "qwerty", "abc123", "changeme", "labelox", "labeloxav",
}


class IdentityError(Exception):
    """A credential operation refused. The message is safe to show a user."""


# ---------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    """scrypt with a fresh salt, returned as a self-describing string.

    The parameters travel with the hash so raising the cost later does not invalidate existing rows: an old
    hash still verifies under its own parameters and can be re-hashed on the next successful sign-in.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
                         dklen=_KEY_BYTES, maxmem=64 * 1024 * 1024)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time verify. A malformed or absent hash is a refusal, never an exception."""
    if not stored:
        return False
    try:
        scheme, n, r, p, salt_b64, key_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = _unb64(salt_b64), _unb64(key_b64)
        got = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p),
                             dklen=len(expected), maxmem=64 * 1024 * 1024)
    except Exception:  # noqa: BLE001 - a corrupt hash must refuse, not 500
        return False
    return hmac.compare_digest(got, expected)


def check_password_policy(password: str, *, name: str | None = None) -> None:
    """Refuse a password that would not survive a dictionary attack. Raises IdentityError with the reason.

    Length is the dominant factor, so the floor is 12 rather than an assortment of character-class rules
    that mostly produce Password1! and a sticky note.
    """
    if len(password) < MIN_PASSWORD_LEN:
        raise IdentityError(f"password must be at least {MIN_PASSWORD_LEN} characters")
    if len(password) > 256:
        # Bounded because scrypt hashes whatever it is given and a megabyte password is a cheap way to
        # make every sign-in expensive.
        raise IdentityError("password must be at most 256 characters")
    lowered = password.lower()
    if lowered in _OBVIOUS or any(w in lowered for w in _OBVIOUS):
        raise IdentityError("password contains a well-known phrase; choose something less guessable")
    if name and name.lower() in lowered:
        raise IdentityError("password must not contain your user name")
    if len(set(password)) < 5:
        raise IdentityError("password repeats too few distinct characters")


async def get_or_create_credential(db: AsyncSession, user_id: uuid.UUID) -> UserCredential:
    cred = await db.get(UserCredential, user_id)
    if cred is None:
        cred = UserCredential(user_id=user_id, recovery_hashes=[])
        db.add(cred)
        await db.flush()
    return cred


async def set_password(db: AsyncSession, user: User, password: str) -> None:
    check_password_policy(password, name=user.name)
    cred = await get_or_create_credential(db, user.user_id)
    cred.password_hash = hash_password(password)
    cred.password_set_at = datetime.now(UTC)
    # A password change ends every existing session. Otherwise a user who changes it because they believe
    # it was stolen has not actually evicted whoever stole it.
    user.token_version = int(user.token_version) + 1
    cred.failed_attempts = 0
    cred.locked_until = None
    await db.commit()
    log.info("identity.password_set", user=str(user.user_id))


def lockout_remaining(cred: UserCredential | None, now: datetime | None = None) -> int:
    """Seconds left on a lockout, 0 if not locked."""
    if cred is None or cred.locked_until is None:
        return 0
    now = now or datetime.now(UTC)
    return max(0, int((cred.locked_until - now).total_seconds()))


async def record_failure(db: AsyncSession, cred: UserCredential) -> None:
    """Count a failed attempt and lock the account once the ceiling is reached."""
    cred.failed_attempts = int(cred.failed_attempts or 0) + 1
    if cred.failed_attempts >= MAX_FAILED_ATTEMPTS:
        cred.locked_until = datetime.now(UTC) + timedelta(minutes=LOCKOUT_MINUTES)
        cred.failed_attempts = 0
        log.warning("identity.locked_out", user=str(cred.user_id))
    await db.commit()


async def record_success(db: AsyncSession, cred: UserCredential) -> None:
    cred.failed_attempts = 0
    cred.locked_until = None
    await db.commit()


# ---------------------------------------------------------------- TOTP (RFC 6238)

def new_totp_secret() -> str:
    """A 160-bit base32 secret, which is what authenticator apps expect."""
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def totp_uri(secret: str, account: str, issuer: str = "LabeloxAV") -> str:
    """The otpauth:// URI an authenticator app scans."""
    from urllib.parse import quote

    return (f"otpauth://totp/{quote(issuer)}:{quote(account)}"
            f"?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30")


def totp_step(now: float | None = None, period: int = 30) -> int:
    return int((now if now is not None else time.time()) // period)


def totp_code(secret: str, step: int) -> str:
    """The six-digit code for a given time step. RFC 4226 truncation over HMAC-SHA1."""
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    mac = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    code = struct.unpack(">I", mac[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{code % 1_000_000:06d}"


def verify_totp(secret: str, code: str, *, now: float | None = None, window: int = 1,
                last_step: int | None = None) -> int | None:
    """Return the step the code matched, or None.

    The step is returned rather than a bool so the caller can persist it: a code stays valid for a whole
    window, and without recording the last accepted step an attacker who observes one has the remainder of
    that window to replay it.
    """
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit() or len(code) != 6:
        return None
    current = totp_step(now)
    for delta in range(-window, window + 1):
        step = current + delta
        if last_step is not None and step <= last_step:
            continue  # already used, or older than one already used
        if hmac.compare_digest(totp_code(secret, step), code):
            return step
    return None


# ---------------------------------------------------------------- recovery codes

def new_recovery_codes(n: int = RECOVERY_CODES) -> list[str]:
    """Human-transcribable single-use codes. Shown once; only their hashes are kept."""
    return ["-".join(secrets.token_hex(2) for _ in range(3)) for _ in range(n)]


def hash_recovery(code: str) -> str:
    """SHA-256 is right here where a password hash is not: these are 96 bits of true randomness, so there is
    no dictionary to attack and no reason to pay scrypt's cost on every attempt."""
    return hashlib.sha256(code.strip().lower().encode()).hexdigest()


def consume_recovery(cred: UserCredential, code: str) -> bool:
    """Spend a recovery code. Removes it, so it cannot be used twice."""
    want = hash_recovery(code)
    remaining = list(cred.recovery_hashes or [])
    for i, h in enumerate(remaining):
        if hmac.compare_digest(h, want):
            remaining.pop(i)
            cred.recovery_hashes = remaining
            return True
    return False


# ---------------------------------------------------------------- password reset

async def create_reset(db: AsyncSession, user: User) -> str:
    """Issue a single-use reset token and return the plaintext, which is never stored.

    The caller delivers it out of band. Nothing here emails: a deployment's mail path is its own, and
    inventing one would mean either a hard SMTP dependency or a silent no-op that looks like it worked.
    """
    token = secrets.token_urlsafe(32)
    db.add(PasswordReset(user_id=user.user_id, token_hash=hashlib.sha256(token.encode()).hexdigest(),
                         expires_at=datetime.now(UTC) + timedelta(minutes=RESET_TTL_MINUTES)))
    await db.commit()
    log.info("identity.reset_issued", user=str(user.user_id))
    return token


async def consume_reset(db: AsyncSession, token: str, new_password: str) -> User:
    """Spend a reset token and set the new password. Raises IdentityError on anything wrong with it."""
    digest = hashlib.sha256((token or "").encode()).hexdigest()
    row = (await db.execute(
        select(PasswordReset).where(PasswordReset.token_hash == digest))).scalar_one_or_none()
    if row is None or row.used_at is not None:
        raise IdentityError("this reset link is not valid")
    expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
    if expires < datetime.now(UTC):
        raise IdentityError("this reset link has expired")

    user = await db.get(User, row.user_id)
    if user is None:
        raise IdentityError("this reset link is not valid")
    row.used_at = datetime.now(UTC)
    await set_password(db, user, new_password)   # commits, and bumps token_version
    return user


# ---------------------------------------------------------------- OIDC

async def link_or_create_oidc_user(db: AsyncSession, *, issuer: str, subject: str,
                                   email: str | None, preferred_name: str | None,
                                   default_role: str = "annotator") -> tuple[User, bool]:
    """Resolve a directory identity to a local user, creating one on first sign-in.

    Matching is on issuer+subject. Email is recorded but never matched on, because a directory can reassign
    an address and matching on it would hand the new holder the previous person's account and role.
    """
    cred = (await db.execute(
        select(UserCredential).where(UserCredential.oidc_issuer == issuer,
                                     UserCredential.oidc_subject == subject))).scalar_one_or_none()
    if cred is not None:
        user = await db.get(User, cred.user_id)
        if user is not None:
            if email and cred.email != email:
                cred.email = email
                await db.commit()
            return user, False

    base = _safe_name(preferred_name or (email.split("@")[0] if email else "") or f"user-{subject[:8]}")
    name = base
    for i in range(1, 50):
        if (await db.execute(select(User).where(User.name == name))).scalar_one_or_none() is None:
            break
        name = f"{base}-{i}"

    user = User(name=name, role=default_role)
    db.add(user)
    await db.flush()
    db.add(UserCredential(user_id=user.user_id, oidc_issuer=issuer, oidc_subject=subject,
                          email=email, recovery_hashes=[]))
    await db.commit()
    log.info("identity.oidc_user_created", user=str(user.user_id), issuer=issuer)
    return user, True


def _safe_name(raw: str) -> str:
    keep = [c for c in (raw or "").strip().lower() if c.isalnum() or c in "-._"]
    return ("".join(keep) or "user")[:48]


# ---------------------------------------------------------------- helpers

def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def public_state(user: User, cred: UserCredential | None) -> dict:
    """What the profile page is allowed to know. No hash, no secret, no recovery code."""
    return {
        "user_id": str(user.user_id),
        "name": user.name,
        "role": user.role,
        "email": cred.email if cred else None,
        "has_password": bool(cred and cred.password_hash),
        "mfa_enabled": bool(cred and cred.totp_confirmed_at),
        "sso": bool(cred and cred.oidc_issuer),
        "sso_issuer": cred.oidc_issuer if cred else None,
        "recovery_codes_left": len(cred.recovery_hashes or []) if cred else 0,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def json_safe(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
