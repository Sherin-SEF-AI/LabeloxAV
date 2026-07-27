"""Section 2.1: v2 tokens expire and are revocable, and 2.4: the prod guard refuses an unsafe auth posture.
Pure unit tests (no DB): the token layer is stateless, and the config guard is a model validator."""
from __future__ import annotations

import uuid

import pytest

from services.api.auth_token import _b64, _sig, mint_token, verify_token


def test_v2_roundtrip_carries_uid_and_version():
    uid = str(uuid.uuid4())
    tok = mint_token(uid, "key", token_version=3, ttl_seconds=100, now=1000)
    p = verify_token(tok, "key", now=1050)
    assert p is not None and p.uid == uid and p.token_version == 3 and not p.legacy


def test_expired_token_is_rejected():
    tok = mint_token(str(uuid.uuid4()), "key", ttl_seconds=100, now=1000)
    assert verify_token(tok, "key", now=1101) is None      # 1 second past expiry
    assert verify_token(tok, "key", now=1050) is not None   # still valid


def test_wrong_key_is_rejected():
    tok = mint_token(str(uuid.uuid4()), "key", now=1000)
    assert verify_token(tok, "other-key", now=1000) is None


def test_tampered_payload_is_rejected_before_parse():
    uid = str(uuid.uuid4())
    tok = mint_token(uid, "key", token_version=1, ttl_seconds=100, now=1000)
    prefix, _payload, sig = tok.split(".")
    forged = _b64(b'{"uid":"' + uid.encode() + b'","iat":0,"exp":9999999999,"tv":99}')
    assert verify_token(f"{prefix}.{forged}.{sig}", "key", now=1000) is None  # sig no longer matches


def test_legacy_rejected_by_default_accepted_when_enabled():
    uid = str(uuid.uuid4())
    legacy = f"lbx1.{_b64(uid.encode())}.{_b64(_sig(uid.encode(), 'key'))}"
    assert verify_token(legacy, "key") is None                       # off by default
    p = verify_token(legacy, "key", accept_legacy=True)
    assert p is not None and p.uid == uid and p.legacy and p.token_version is None


def test_revocation_is_a_version_mismatch():
    # A token minted at tv=1 carries tv=1; a user whose token_version has advanced to 2 no longer matches, which
    # is exactly the comparison deps.current_user makes.
    uid = str(uuid.uuid4())
    p = verify_token(mint_token(uid, "key", token_version=1, now=1000), "key", now=1000)
    assert p.token_version == 1  # deps rejects when this != user.token_version (2 after a revoke)


def test_prod_guard_refuses_auth_disabled_off_local(monkeypatch):
    monkeypatch.setenv("LBX_ENV", "prod")
    monkeypatch.setenv("LBX_AUTH__ENABLED", "false")
    from core.config import Settings
    with pytest.raises(ValueError, match="auth is disabled"):
        Settings()


def test_prod_guard_refuses_legacy_tokens_off_local(monkeypatch):
    monkeypatch.setenv("LBX_ENV", "prod")
    monkeypatch.setenv("LBX_AUTH__ENABLED", "true")   # so we reach the legacy check, not the auth-off one
    monkeypatch.setenv("LBX_AUTH__ACCEPT_LEGACY_TOKENS", "true")
    monkeypatch.setenv("LBX_AUTH__SIGNING_KEY", "a-strong-non-default-signing-key-xxxxxxxxxx")
    from core.config import Settings
    with pytest.raises(ValueError, match="legacy tokens"):
        Settings()
