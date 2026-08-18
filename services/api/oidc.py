"""An OIDC authorization-code client with PKCE, written against the spec rather than a library.

A dependency here would be reasonable, but the whole flow is a discovery fetch, a redirect, a token
exchange, and a JWT signature check against the provider's JWKS. Writing it directly means the deployment
carries no extra transitive dependency in its auth path, and every check that matters is visible in one
file rather than behind a configuration surface.

Everything the spec calls a security requirement is enforced rather than optional:

- **PKCE (S256)** on every request, so an intercepted code cannot be exchanged without the verifier.
- **state**, bound to the browser, so a code cannot be injected from another session.
- **nonce**, bound into the id_token, so a token minted for a different login cannot be replayed here.
- **Signature verified against the provider's JWKS**, with the key selected by `kid`. An unsigned or
  `alg: none` token is refused, which is the classic JWT bypass.
- **iss, aud, exp, iat checked.** A token that verifies cryptographically but was issued for a different
  audience is still not ours.

RS256 is implemented directly (PKCS#1 v1.5 over SHA-256) because it is what essentially every provider
uses; anything else is refused by name rather than silently accepted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.logging import get_logger
from services.api.ephemeral import EphemeralStore

log = get_logger("oidc")

_DISCOVERY_TTL = 3600
_JWKS_TTL = 3600


class OidcError(Exception):
    """A federated sign-in refused. The message is safe to show a user."""


@dataclass
class _Cached:
    value: Any = None
    at: float = 0.0

    def fresh(self, ttl: float) -> bool:
        return self.value is not None and (time.time() - self.at) < ttl


_discovery: dict[str, _Cached] = {}
_jwks: dict[str, _Cached] = {}


@dataclass
class PendingLogin:
    """The per-attempt secrets, held server-side and keyed by an opaque handle in a cookie.

    Held here rather than in the cookie itself so the verifier never reaches the browser: a cookie the user
    agent can read is a cookie an injected script can read, and PKCE would then protect nothing.
    """

    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    created_at: float = field(default_factory=time.time)
    next_url: str = "/"


_PENDING_TTL = 600
# Shared across workers when Redis answers. This was a module-level dict, and the OIDC callback is a
# *browser redirect* - it can land on any worker, not the one that started the login. Under more than one
# worker an (N-1)/N share of SSO sign-ins failed with "this sign-in attempt has expired", which looks like
# a provider problem and is not one. The verifier still never reaches the browser, which is the property
# the dict was there for.
_pending = EphemeralStore("oidc", _PENDING_TTL)


async def discover(issuer: str, *, client: httpx.AsyncClient | None = None) -> dict:
    """Fetch and cache the provider's configuration document."""
    issuer = issuer.rstrip("/")
    hit = _discovery.get(issuer)
    if hit and hit.fresh(_DISCOVERY_TTL):
        return hit.value
    url = f"{issuer}/.well-known/openid-configuration"
    owns = client is None
    client = client or httpx.AsyncClient(timeout=10.0)
    try:
        r = await client.get(url)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:  # noqa: BLE001
        raise OidcError(f"could not reach the identity provider at {issuer}") from exc
    finally:
        if owns:
            await client.aclose()
    if doc.get("issuer", "").rstrip("/") != issuer:
        # The document must claim the issuer we asked for, or a redirect could have landed us on someone
        # else's provider and we would trust their tokens.
        raise OidcError("the provider's configuration does not match its issuer")
    _discovery[issuer] = _Cached(doc, time.time())
    return doc


async def fetch_jwks(jwks_uri: str, *, client: httpx.AsyncClient | None = None) -> dict:
    hit = _jwks.get(jwks_uri)
    if hit and hit.fresh(_JWKS_TTL):
        return hit.value
    owns = client is None
    client = client or httpx.AsyncClient(timeout=10.0)
    try:
        r = await client.get(jwks_uri)
        r.raise_for_status()
        keys = r.json()
    except Exception as exc:  # noqa: BLE001
        raise OidcError("could not fetch the provider's signing keys") from exc
    finally:
        if owns:
            await client.aclose()
    _jwks[jwks_uri] = _Cached(keys, time.time())
    return keys


async def begin(issuer_doc: dict, *, client_id: str, redirect_uri: str, scopes: str,
                next_url: str = "/") -> tuple[str, str]:
    """Start a login. Returns (authorization_url, handle); the handle goes in a short-lived cookie."""
    verifier = _b64(secrets.token_bytes(48))
    challenge = _b64(hashlib.sha256(verifier.encode()).digest())
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    handle = secrets.token_urlsafe(24)
    await _pending.put(handle, {"state": state, "nonce": nonce, "verifier": verifier,
                                "redirect_uri": redirect_uri, "next_url": next_url,
                                "created_at": time.time()})

    from urllib.parse import urlencode

    q = urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    return f"{issuer_doc['authorization_endpoint']}?{q}", handle


async def take_pending(handle: str | None) -> PendingLogin:
    d = await _pending.take(handle)   # one handle, one attempt; read and delete are atomic
    if d is None:
        raise OidcError("this sign-in attempt has expired; start again")
    return PendingLogin(**d)


async def exchange(issuer_doc: dict, *, code: str, pending: PendingLogin, client_id: str,
                   client_secret: str | None, client: httpx.AsyncClient | None = None) -> dict:
    """Exchange the code for tokens, presenting the PKCE verifier."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending.redirect_uri,
        "client_id": client_id,
        "code_verifier": pending.verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    owns = client is None
    client = client or httpx.AsyncClient(timeout=15.0)
    try:
        r = await client.post(issuer_doc["token_endpoint"], data=data,
                              headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise OidcError(f"the identity provider refused the sign-in ({r.status_code})")
        return r.json()
    except OidcError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OidcError("could not complete the exchange with the identity provider") from exc
    finally:
        if owns:
            await client.aclose()


def verify_id_token(token: str, *, jwks: dict, issuer: str, audience: str, nonce: str,
                    now: float | None = None, leeway: int = 120) -> dict:
    """Verify an id_token end to end and return its claims.

    Refuses before parsing anything that matters: an unsigned token, an unknown algorithm, or a key the
    provider does not publish. Those are the bypasses that actually get exploited.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise OidcError("the identity token is malformed")
    header = json.loads(_unb64(parts[0]))
    alg = header.get("alg")
    if alg != "RS256":
        # Named refusal rather than a quiet fallback. "none" is the classic bypass and every other
        # algorithm here would be unverified.
        raise OidcError(f"unsupported identity-token algorithm {alg!r}")

    key = _select_key(jwks, header.get("kid"))
    signing_input = f"{parts[0]}.{parts[1]}".encode()
    if not _rs256_verify(signing_input, _unb64(parts[2]), key):
        raise OidcError("the identity token signature does not verify")

    claims = json.loads(_unb64(parts[1]))
    now = now if now is not None else time.time()
    if str(claims.get("iss", "")).rstrip("/") != issuer.rstrip("/"):
        raise OidcError("the identity token was issued by a different provider")
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    if audience not in auds:
        raise OidcError("the identity token was issued for a different application")
    if float(claims.get("exp", 0)) < now - leeway:
        raise OidcError("the identity token has expired")
    if float(claims.get("iat", 0)) > now + leeway:
        raise OidcError("the identity token is not yet valid")
    if claims.get("nonce") != nonce:
        # Without this a token minted for another login of the same user could be replayed into this one.
        raise OidcError("the identity token does not match this sign-in attempt")
    if not claims.get("sub"):
        raise OidcError("the identity token carries no subject")
    return claims


def _select_key(jwks: dict, kid: str | None) -> dict:
    keys = [k for k in (jwks or {}).get("keys", []) if k.get("kty") == "RSA"]
    if kid:
        for k in keys:
            if k.get("kid") == kid:
                return k
    if len(keys) == 1:
        return keys[0]
    raise OidcError("the provider does not publish the key this token was signed with")


# PKCS#1 v1.5 over SHA-256. The DigestInfo prefix is fixed for SHA-256 and is part of the padding the
# signer produced, so comparing the whole reconstructed block is the verification.
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def _rs256_verify(message: bytes, signature: bytes, jwk: dict) -> bool:
    try:
        n = int.from_bytes(_unb64(jwk["n"]), "big")
        e = int.from_bytes(_unb64(jwk["e"]), "big")
        sig = int.from_bytes(signature, "big")
        if sig >= n:
            return False
        k = (n.bit_length() + 7) // 8
        decoded = pow(sig, e, n).to_bytes(k, "big")
        expected = (b"\x00\x01" + b"\xff" * (k - len(_SHA256_DIGEST_INFO) - 35)
                    + b"\x00" + _SHA256_DIGEST_INFO + hashlib.sha256(message).digest())
        return secrets.compare_digest(decoded, expected)
    except Exception:  # noqa: BLE001 - any malformed key or signature is a refusal
        return False


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
