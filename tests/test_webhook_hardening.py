"""Webhook delivery: SSRF refusal, replay-resistant signatures, and retry on transient failure.

Three defects:

1. A subscription URL was validated only as "starts with http", so the server could be pointed at
   169.254.169.254 (cloud instance credentials) or localhost:9000 (MinIO) and would fetch it with its own
   network position. That is a server-side request forgery primitive handed to any authenticated caller.
2. The signature covered the body alone, so a captured delivery replayed verbatim verified forever.
3. Delivery was a single POST. One timeout or one 502 and the event was gone: no retry, no record.

Pure unit tests: DNS resolution for the SSRF check is real (it must be, since the point is to resolve names
before trusting them), but no HTTP is sent and no database is touched."""
from __future__ import annotations

import pytest

from core.config import get_settings
from services.integrations.webhooks import (
    _is_safe_webhook_url,
    _should_retry,
    sign,
    verify,
)


@pytest.fixture
def ssrf_guard_on():
    """The suite opts into private targets globally (its receivers are on localhost), so the tests that
    exercise the guard itself have to turn it back on."""
    s = get_settings()
    prev = s.integrations.allow_private_webhook_targets
    s.integrations.allow_private_webhook_targets = False
    yield
    s.integrations.allow_private_webhook_targets = prev


# ---- SSRF ----

def test_cloud_metadata_endpoint_is_refused(ssrf_guard_on):
    ok, why = _is_safe_webhook_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
    assert not ok and "non-public" in why


def test_loopback_and_private_ranges_are_refused(ssrf_guard_on):
    for url in ("http://127.0.0.1:9000/bucket", "http://10.0.0.5/hook",
                "http://192.168.1.10/hook", "http://172.16.0.1/hook"):
        ok, _ = _is_safe_webhook_url(url)
        assert not ok, f"{url} must be refused"


def test_non_http_schemes_are_refused(ssrf_guard_on):
    for url in ("file:///etc/passwd", "gopher://evil/", "ftp://host/x"):
        assert not _is_safe_webhook_url(url)[0]


def test_a_public_https_endpoint_is_allowed(ssrf_guard_on):
    assert _is_safe_webhook_url("https://example.com/hooks/labelox")[0]


def test_unresolvable_host_is_deferred_to_delivery_time(ssrf_guard_on):
    # Unresolvable now is not proof of danger (DNS may not have propagated), and the authoritative check runs
    # again immediately before the fetch, which is also what closes DNS rebinding.
    assert _is_safe_webhook_url("https://this-host-does-not-exist.invalid/hook")[0]


# ---- signature and replay ----

def test_timestamped_signature_verifies_inside_the_window():
    body = b'{"event":"model.promoted"}'
    sig = sign("secret", body, 1_700_000_000)
    assert verify("secret", body, sig, now=1_700_000_060)


def test_replayed_delivery_outside_the_window_is_rejected():
    # The signature is still arithmetically correct; it is the age that fails. This is the whole point.
    body = b'{"event":"model.promoted"}'
    sig = sign("secret", body, 1_700_000_000)
    assert not verify("secret", body, sig, now=1_700_100_000)


def test_tampered_body_fails_even_with_a_fresh_timestamp():
    sig = sign("secret", b'{"amount":1}', 1_700_000_000)
    assert not verify("secret", b'{"amount":9999}', sig, now=1_700_000_010)


def test_wrong_secret_fails():
    body = b"payload"
    assert not verify("other", body, sign("secret", body, 1_700_000_000), now=1_700_000_010)


def test_legacy_unsigned_timestamp_form_still_verifies():
    # Receivers written against the previous scheme must keep working.
    body = b"payload"
    assert verify("secret", body, sign("secret", body))


def test_malformed_signature_is_rejected_not_crashed():
    for bad in ("", "t=notanumber,sha256=abc", "t=", "garbage"):
        assert not verify("secret", b"x", bad)


# ---- retry policy ----

def test_transport_failure_and_server_errors_are_retried():
    assert _should_retry(None)      # timeout, DNS, connection refused
    assert _should_retry(500)
    assert _should_retry(503)
    assert _should_retry(429)       # explicit "try again"


def test_client_errors_are_not_retried():
    # The receiver understood and rejected it; repeating changes nothing and only amplifies load.
    for status in (400, 401, 403, 404, 422):
        assert not _should_retry(status)


def test_success_is_not_retried():
    for status in (200, 201, 204):
        assert not _should_retry(status)
