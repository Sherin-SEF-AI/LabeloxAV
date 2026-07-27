"""Stateless signed API tokens (C1). A token binds a user under an HMAC over the server signing key, so a
client can neither forge one nor self-assert a different user or role: the server verifies the signature, then
loads the authoritative role from the DB row. This replaces the old plaintext X-Lbx-User-Id header, which let
anyone who read the public user list act as any user (including admin) by echoing that user's UUID.

Format (v2): "lbx2.<b64url(json_payload)>.<b64url(hmac_sha256(json_payload, signing_key))>" where the payload
is {"uid", "iat", "exp", "tv"}. Two properties the old lbx1 format lacked:

* Expiry (exp): a leaked token is valid for at most token_ttl_seconds, not forever.
* Revocation (tv, a token version): the verifier compares it against User.token_version, so incrementing that
  one column invalidates every token for that user at once, with no session store and without rotating the
  global key. The tv check needs the DB row, so it happens in the caller (deps.current_user), which already
  loads the user for its role; verify_token returns the payload including tv for that comparison.

The signature is verified BEFORE the payload JSON is parsed, so a malformed or hostile payload can never reach
the parser unauthenticated. Legacy lbx1 tokens are accepted only when accept_legacy_tokens is set (off by
default, refused off-local), with a structured warning on every verify, so the old format can be retired.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from uuid import UUID

from core.logging import get_logger

log = get_logger("auth_token")

_PREFIX = "lbx2"
_LEGACY_PREFIX = "lbx1"


@dataclass(frozen=True)
class TokenPayload:
    uid: str
    token_version: int | None      # None for a legacy token, which cannot be revoked
    exp: int | None
    legacy: bool = False


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sig(msg: bytes, key: str) -> bytes:
    return hmac.new(key.encode(), msg, hashlib.sha256).digest()


def mint_token(user_id: str | UUID, signing_key: str, *, token_version: int = 1,
               ttl_seconds: int = 43200, now: int | None = None) -> str:
    """Issue a signed v2 token for a user. Called only from server-side, role-gated issuance paths, so a token
    can only originate from an already-authorized actor."""
    now = int(now if now is not None else time.time())
    payload = json.dumps({"uid": str(user_id), "iat": now, "exp": now + int(ttl_seconds),
                          "tv": int(token_version)}, separators=(",", ":"), sort_keys=True).encode()
    return f"{_PREFIX}.{_b64(payload)}.{_b64(_sig(payload, signing_key))}"


def _verify_legacy(parts: list[str], signing_key: str) -> TokenPayload | None:
    """Verify an lbx1 token: HMAC over the raw user_id, no expiry, no revocation."""
    try:
        uid = _unb64(parts[1]).decode()
        sig = _unb64(parts[2])
    except Exception:  # noqa: BLE001
        return None
    if not hmac.compare_digest(sig, _sig(uid.encode(), signing_key)):
        return None
    try:
        UUID(uid)
    except ValueError:
        return None
    log.warning("auth_token.legacy_verified", uid=uid)
    return TokenPayload(uid=uid, token_version=None, exp=None, legacy=True)


def verify_token(token: str | None, signing_key: str, *, accept_legacy: bool = False,
                 now: int | None = None) -> TokenPayload | None:
    """Return the token payload if the signature matches and the token has not expired, else None. The signature
    is checked before the payload JSON is parsed. The token_version is returned, not checked here (the caller
    compares it against the user row)."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None

    if parts[0] == _LEGACY_PREFIX:
        return _verify_legacy(parts, signing_key) if accept_legacy else None
    if parts[0] != _PREFIX:
        return None

    try:
        raw = _unb64(parts[1])
        sig = _unb64(parts[2])
    except Exception:  # noqa: BLE001
        return None
    # Verify the signature over the RAW payload bytes BEFORE parsing the JSON.
    if not hmac.compare_digest(sig, _sig(raw, signing_key)):
        return None
    try:
        data = json.loads(raw)
        uid = str(data["uid"])
        UUID(uid)
        exp = int(data["exp"])
        tv = int(data["tv"])
    except Exception:  # noqa: BLE001 - a malformed but correctly-signed payload is still rejected
        return None
    if exp < int(now if now is not None else time.time()):
        return None
    return TokenPayload(uid=uid, token_version=tv, exp=exp, legacy=False)


def bearer_payload(authorization: str | None, signing_key: str, *, accept_legacy: bool = False,
                   now: int | None = None) -> TokenPayload | None:
    """Extract and verify the payload from an 'Authorization: Bearer <token>' header value."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return verify_token(token.strip(), signing_key, accept_legacy=accept_legacy, now=now)


def bearer_uid(authorization: str | None, signing_key: str, *, accept_legacy: bool = False) -> str | None:
    """The verified user_id from a Bearer header, or None. Does NOT check token_version (revocation): callers
    that can load the user row should use bearer_payload and compare tv. Kept for paths without DB access."""
    payload = bearer_payload(authorization, signing_key, accept_legacy=accept_legacy)
    return payload.uid if payload else None
