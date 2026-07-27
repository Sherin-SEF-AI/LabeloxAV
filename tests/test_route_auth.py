"""Section 2.2/2.3: the read allowlist stays minimal and fail-closed, enforced over the real route table.

Pure unit test (no DB): it inspects the mounted routes and the middleware classifier, so it runs in the fast
suite and guards the invariant on every commit. The infra-gated behavioural proof (401 unauth, 403
under-privileged over live requests) lives in test_auth.py."""
from __future__ import annotations

from services.api.main import (
    _APPROVED_PUBLIC_READ_PREFIXES as APPROVED_PUBLIC_READ_PREFIXES,
)
from services.api.main import (
    _assert_auth_floors,
    _is_public_read,
    _required_role,
    app,
)


def _api_get_paths() -> list[str]:
    paths: list[str] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if path.startswith("/api/") and ("GET" in methods or "HEAD" in methods):
            paths.append(path)
    return paths


def test_only_health_is_a_public_read():
    # Every mounted GET/HEAD under /api/ that the middleware treats as public must be in the approved set.
    # A new read route defaults to gated; making one public trips this test until it is approved here.
    leaked = [p for p in _api_get_paths()
              if _is_public_read(p) and not p.startswith(APPROVED_PUBLIC_READ_PREFIXES)]
    assert leaked == [], f"read routes public without approval: {leaked}"


def test_sensitive_reads_are_gated():
    # The routes the denylist used to name explicitly (data exfiltration and info disclosure) must not be
    # public under the inverted default. Before the inversion /api/ontology and the rest were open.
    for path in ("/api/ontology", "/api/datasets", "/api/govern/state", "/api/users", "/api/export"):
        assert not _is_public_read(path), f"{path} must require a token to read"


def test_health_stays_public():
    assert _is_public_read("/api/health")


def test_startup_backstop_passes_on_the_real_app():
    # The lifespan handler runs this before serving; it must accept the real route table and inspect a
    # non-trivial number of /api routes (a zero would mean the check silently found nothing to guard).
    checked = _assert_auth_floors(app)
    assert checked > 100


def test_role_floors_are_sane():
    # Spot-check the floor map: governance and user administration are admin, corpus mutation is reviewer,
    # and an unclassified route still requires at least an authenticated annotator (never anonymous).
    assert _required_role("/api/govern/promote") == "admin"
    assert _required_role("/api/users") == "admin"
    assert _required_role("/api/export") == "reviewer"
    assert _required_role("/api/objects/x/bulk-review") == "reviewer"
    assert _required_role("/api/users/me") == "annotator"  # self-service, not the admin floor of its prefix
    assert _required_role("/api/segment") == "annotator"   # unclassified: authenticated, never anonymous
