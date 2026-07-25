"""Stateless signed API tokens (C1). A token binds a user_id under an HMAC over the server signing key, so a
client can neither forge one nor self-assert a different user or role: the server verifies the signature, then
loads the authoritative role from the DB row. This replaces the old plaintext X-Lbx-User-Id header, which let
anyone who read the public user list act as any user (including admin) simply by echoing that user's UUID.

Format: "lbx1.<b64url(user_id)>.<b64url(hmac_sha256(user_id, signing_key))>". Stateless (no DB column, no
migration): the signing key is the only secret, held server-side, and mints/verifies every token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from uuid import UUID

_PREFIX = "lbx1"


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sig(user_id: str, key: str) -> bytes:
    return hmac.new(key.encode(), user_id.encode(), hashlib.sha256).digest()


def mint_token(user_id: str | UUID, signing_key: str) -> str:
    """Issue a signed token for a user. Called only from server-side issuance paths (user create / re-issue),
    both of which are role-gated, so a token can only originate from an already-authorized actor."""
    uid = str(user_id)
    return f"{_PREFIX}.{_b64(uid.encode())}.{_b64(_sig(uid, signing_key))}"


def verify_token(token: str | None, signing_key: str) -> str | None:
    """Return the user_id if the token is well-formed and its signature matches, else None. Uses a
    constant-time compare so the signature cannot be recovered by timing."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _PREFIX:
        return None
    try:
        uid = _unb64(parts[1]).decode()
        sig = _unb64(parts[2])
    except Exception:  # noqa: BLE001
        return None
    if not hmac.compare_digest(sig, _sig(uid, signing_key)):
        return None
    try:
        UUID(uid)
    except ValueError:
        return None
    return uid


def bearer_uid(authorization: str | None, signing_key: str) -> str | None:
    """Extract and verify a user_id from an 'Authorization: Bearer <token>' header value."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return verify_token(token.strip(), signing_key)
