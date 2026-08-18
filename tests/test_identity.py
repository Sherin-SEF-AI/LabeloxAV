"""Passwords, second factors, federated identity, and the self-service routes around them.

Before this the only credential was an admin-minted token: a person could not obtain access themselves, add
a second factor, or sign in through a directory. That was the largest single blocker to deploying this
anywhere real.

Most of what is asserted here is about refusal rather than success, because an authentication system is
defined by what it declines. The two properties defended hardest:

- **The login form must not enumerate accounts.** A real and an absent user get the same answer, and both
  pay for a password hash so the timing does not answer the question the message refused to.
- **A first factor alone grants nothing.** The MFA challenge is short-lived, single-use, and can do nothing
  but finish the one sign-in it belongs to.
"""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from services.api import identity as ident

pytestmark = pytest.mark.db


def _client() -> TestClient:
    from _authutil import _clear_db_cache

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


def _uniq(prefix: str = "u") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


GOOD = "correct-horse-battery-staple-7"


# ---------------------------------------------------------------- hashing

def test_a_password_verifies_only_against_itself():
    h = ident.hash_password(GOOD)
    assert ident.verify_password(GOOD, h)
    assert not ident.verify_password(GOOD + "x", h)
    assert not ident.verify_password("", h)


def test_two_hashes_of_the_same_password_differ():
    """A fresh salt per hash, so identical passwords are not identifiable as identical from the table."""
    assert ident.hash_password(GOOD) != ident.hash_password(GOOD)


def test_a_corrupt_hash_refuses_rather_than_raising():
    """A damaged row must fail closed. Raising here would turn a bad record into a 500 on the login path."""
    for bad in ("", "not-a-hash", "scrypt$x$y$z$q$r", "bcrypt$1$2$3$4$5"):
        assert ident.verify_password(GOOD, bad) is False


def test_the_password_policy_refuses_the_guessable_ones():
    for bad, why in [("short", "too short"), ("passwordpassword", "well-known phrase"),
                     ("aaaaaaaaaaaaaaaa", "too few distinct")]:
        with pytest.raises(ident.IdentityError):
            ident.check_password_policy(bad)
        assert why  # each case is here for a stated reason
    with pytest.raises(ident.IdentityError):
        ident.check_password_policy("alice-alice-alice", name="alice")
    ident.check_password_policy(GOOD, name="alice")   # and a good one passes


# ---------------------------------------------------------------- TOTP

def test_a_totp_code_verifies_in_its_window_and_not_outside_it():
    secret = ident.new_totp_secret()
    now = time.time()
    code = ident.totp_code(secret, ident.totp_step(now))
    assert ident.verify_totp(secret, code, now=now) is not None
    # An hour later the same code is meaningless.
    assert ident.verify_totp(secret, code, now=now + 3600) is None


def test_a_totp_code_cannot_be_replayed_inside_its_own_window():
    """A code stays valid for a whole window, so without recording the accepted step an attacker who
    observes one has the rest of that window to reuse it."""
    secret = ident.new_totp_secret()
    now = time.time()
    step = ident.totp_step(now)
    code = ident.totp_code(secret, step)

    accepted = ident.verify_totp(secret, code, now=now)
    assert accepted == step
    assert ident.verify_totp(secret, code, now=now, last_step=accepted) is None


def test_a_malformed_totp_code_is_refused():
    secret = ident.new_totp_secret()
    for bad in ("", "abc", "12345", "1234567", "12 34 56 78"):
        assert ident.verify_totp(secret, bad) is None


def test_a_totp_code_from_another_secret_is_refused():
    a, b = ident.new_totp_secret(), ident.new_totp_secret()
    now = time.time()
    assert ident.verify_totp(a, ident.totp_code(b, ident.totp_step(now)), now=now) is None


def test_a_recovery_code_works_once():
    from db.models import UserCredential

    codes = ident.new_recovery_codes(3)
    cred = UserCredential(user_id=uuid.uuid4(),
                          recovery_hashes=[ident.hash_recovery(c) for c in codes])
    assert ident.consume_recovery(cred, codes[1]) is True
    assert ident.consume_recovery(cred, codes[1]) is False    # spent
    assert len(cred.recovery_hashes) == 2
    assert ident.consume_recovery(cred, "not-a-code") is False


# ---------------------------------------------------------------- sign-in over HTTP

def _signup(c: TestClient, name: str, password: str = GOOD, **kw) -> dict:
    r = c.post("/api/auth/signup", json={"name": name, "password": password, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_the_methods_endpoint_is_reachable_without_a_token():
    """The login page has to render before anyone is signed in, so this one is public by necessity."""
    with _client() as c:
        r = c.get("/api/auth/methods")
    assert r.status_code == 200
    assert set(r.json()) >= {"password", "self_signup", "oidc", "bootstrap"}


def test_signing_up_and_then_in_yields_a_working_token(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        created = _signup(c, name)
        assert created["token"] and created["role"] == "annotator"

        r = c.post("/api/auth/login", json={"name": name, "password": GOOD})
        assert r.status_code == 200
        token = r.json()["token"]
        # The token is a real one, not a placeholder: it opens a gated route.
        assert c.get("/api/auth/profile",
                     headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_a_wrong_password_and_an_absent_account_answer_identically(monkeypatch):
    """The enumeration defence. Two different failures must be indistinguishable to the caller, or the
    login form becomes the account list a credential-stuffing run needs."""
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        _signup(c, name)
        wrong = c.post("/api/auth/login", json={"name": name, "password": "not-the-password-at-all"})
        absent = c.post("/api/auth/login", json={"name": _uniq("ghost"), "password": GOOD})

    assert wrong.status_code == absent.status_code == 401
    assert wrong.json()["detail"] == absent.json()["detail"]


def test_self_signup_is_refused_when_the_operator_has_not_enabled_it(monkeypatch):
    """Open registration on a corpus of personal data is a decision, not a default somebody discovers."""
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", False, raising=False)
    with _client() as c:
        # Not the bootstrap case: the suite has already seeded users.
        r = c.post("/api/auth/signup", json={"name": _uniq(), "password": GOOD})
    assert r.status_code == 403


def test_signup_refuses_a_taken_name_and_a_weak_password(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        _signup(c, name)
        assert c.post("/api/auth/signup",
                      json={"name": name, "password": GOOD}).status_code == 409
        assert c.post("/api/auth/signup",
                      json={"name": _uniq(), "password": "short"}).status_code == 400
        assert c.post("/api/auth/signup",
                      json={"name": "not a valid name!", "password": GOOD}).status_code == 400


def test_repeated_failures_lock_the_account(monkeypatch):
    """In the database, not a cache: a lockout a process restart clears is not a lockout."""
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        _signup(c, name)
        codes = [c.post("/api/auth/login", json={"name": name, "password": "wrong-wrong-wrong"}).status_code
                 for _ in range(ident.MAX_FAILED_ATTEMPTS + 1)]
        assert codes[-1] == 429
        # And the correct password does not bypass the lockout.
        assert c.post("/api/auth/login", json={"name": name, "password": GOOD}).status_code == 429


def test_changing_a_password_requires_the_current_one_and_ends_other_sessions(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name, new = _uniq(), "an-entirely-different-passphrase-2"
    with _client() as c:
        tok = _signup(c, name)["token"]
        h = {"Authorization": f"Bearer {tok}"}

        assert c.post("/api/auth/password/change",
                      json={"current_password": "wrong", "new_password": new},
                      headers=h).status_code == 401

        r = c.post("/api/auth/password/change",
                   json={"current_password": GOOD, "new_password": new}, headers=h)
        assert r.status_code == 200
        fresh = r.json()["token"]

        # The old token is dead (token_version moved), the returned one works, and so does the new password.
        assert c.get("/api/auth/profile", headers=h).status_code == 401
        assert c.get("/api/auth/profile",
                     headers={"Authorization": f"Bearer {fresh}"}).status_code == 200
        assert c.post("/api/auth/login", json={"name": name, "password": new}).status_code == 200


def test_a_reset_token_works_once_and_a_forged_one_never(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name, new = _uniq(), "yet-another-good-passphrase-9"
    with _client() as c:
        _signup(c, name)
        req = c.post("/api/auth/password/reset-request", json={"name": name})
        assert req.status_code == 200
        token = req.json().get("reset_token")     # returned directly on a local deployment
        assert token

        assert c.post("/api/auth/password/reset",
                      json={"token": token, "password": new}).status_code == 200
        # Spent, so a token recovered from a mailbox later is inert.
        assert c.post("/api/auth/password/reset",
                      json={"token": token, "password": new}).status_code == 400
        assert c.post("/api/auth/password/reset",
                      json={"token": "forged", "password": new}).status_code == 400


def test_a_reset_request_for_an_unknown_account_answers_the_same(monkeypatch):
    with _client() as c:
        r = c.post("/api/auth/password/reset-request", json={"name": _uniq("ghost")})
    assert r.status_code == 200 and r.json()["requested"] is True


def test_enrolling_a_second_factor_and_signing_in_with_it(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        tok = _signup(c, name)["token"]
        h = {"Authorization": f"Bearer {tok}"}

        setup = c.post("/api/auth/mfa/setup", headers=h).json()
        secret = setup["secret"]
        assert setup["otpauth_uri"].startswith("otpauth://totp/")

        code = ident.totp_code(secret, ident.totp_step())
        confirmed = c.post("/api/auth/mfa/confirm", json={"code": code}, headers=h)
        assert confirmed.status_code == 200
        recovery = confirmed.json()["recovery_codes"]
        assert len(recovery) == ident.RECOVERY_CODES

        # The password alone now yields a challenge instead of a token.
        first = c.post("/api/auth/login", json={"name": name, "password": GOOD}).json()
        assert first.get("mfa_required") is True and "token" not in first

        # A wrong code does not finish it, and the handle is spent either way.
        assert c.post("/api/auth/login/mfa",
                      json={"mfa_handle": first["mfa_handle"], "code": "000000"}).status_code == 401

        # The NEXT step, not the current one. Confirming enrolment consumed the current step, and the
        # replay guard is right to refuse it a second time: a user in this position waits for the code to
        # roll, which is exactly what stepping forward here models.
        second = c.post("/api/auth/login", json={"name": name, "password": GOOD}).json()
        done = c.post("/api/auth/login/mfa",
                      json={"mfa_handle": second["mfa_handle"],
                            "code": ident.totp_code(secret, ident.totp_step() + 1)})
        assert done.status_code == 200 and done.json()["token"]


def test_a_recovery_code_finishes_a_sign_in_and_is_then_spent(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        tok = _signup(c, name)["token"]
        h = {"Authorization": f"Bearer {tok}"}
        secret = c.post("/api/auth/mfa/setup", headers=h).json()["secret"]
        codes = c.post("/api/auth/mfa/confirm",
                       json={"code": ident.totp_code(secret, ident.totp_step())},
                       headers=h).json()["recovery_codes"]

        challenge = c.post("/api/auth/login", json={"name": name, "password": GOOD}).json()
        assert c.post("/api/auth/login/mfa",
                      json={"mfa_handle": challenge["mfa_handle"], "code": codes[0]}).status_code == 200

        again = c.post("/api/auth/login", json={"name": name, "password": GOOD}).json()
        assert c.post("/api/auth/login/mfa",
                      json={"mfa_handle": again["mfa_handle"], "code": codes[0]}).status_code == 401


async def test_an_mfa_handle_is_single_use():
    """The challenge can finish exactly one sign-in. Reusable, it would be a bearer credential of its own.

    Async because the handle now lives in a store shared across workers rather than a module-level dict:
    the second-factor request lands wherever the load balancer sends it, and a per-process dict failed
    (N-1)/N of MFA sign-ins with a message that reads as an expired handle. Single-use is still the
    property under test, and it is now enforced by an atomic read-and-delete rather than dict.pop.
    """
    from services.api.routers.identity_routes import _mfa_begin, _mfa_take

    handle = await _mfa_begin(uuid.uuid4())
    assert await _mfa_take(handle)
    with pytest.raises(Exception):
        await _mfa_take(handle)


def test_revoking_sessions_invalidates_every_existing_token(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    name = _uniq()
    with _client() as c:
        tok = _signup(c, name)["token"]
        h = {"Authorization": f"Bearer {tok}"}
        fresh = c.post("/api/auth/sessions/revoke", headers=h).json()["token"]

        assert c.get("/api/auth/profile", headers=h).status_code == 401
        assert c.get("/api/auth/profile",
                     headers={"Authorization": f"Bearer {fresh}"}).status_code == 200


def test_the_profile_never_returns_a_secret(monkeypatch):
    from core.config import get_settings

    monkeypatch.setattr(get_settings().auth, "self_signup", True, raising=False)
    with _client() as c:
        tok = _signup(c, _uniq())["token"]
        body = c.get("/api/auth/profile", headers={"Authorization": f"Bearer {tok}"}).json()
    assert body["has_password"] is True and body["mfa_enabled"] is False
    blob = str(body).lower()
    for leak in ("scrypt", "password_hash", "totp_secret", "recovery_hashes"):
        assert leak not in blob


def test_the_credential_routes_are_reachable_unauthenticated_and_nothing_else_is():
    """The allowlist is the security boundary here: a prefix would silently expose whatever gets added
    under /api/auth next, including changing a password or removing a second factor."""
    from services.api.main import _CREDENTIAL_PATHS, _is_credential_path

    assert _is_credential_path("/api/auth/login")
    assert _is_credential_path("/api/auth/methods")
    for gated in ("/api/auth/password/change", "/api/auth/mfa/setup", "/api/auth/mfa/disable",
                  "/api/auth/profile", "/api/auth/sessions/revoke", "/api/auth/refresh"):
        assert not _is_credential_path(gated), f"{gated} must stay behind the token gate"
    assert all(p.startswith("/api/auth/") for p in _CREDENTIAL_PATHS)

    with _client() as c:
        assert c.get("/api/auth/profile").status_code == 401
        assert c.post("/api/auth/mfa/setup").status_code == 401
