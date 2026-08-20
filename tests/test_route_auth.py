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


def test_only_the_probes_stay_public():
    # Both load-balancer probes are public because a probe cannot carry a token. readyz exists because
    # health returns 200 with a degraded body (so an operator sees which dependency is down), which a load
    # balancer would read as healthy and keep routing to a node whose Postgres is gone.
    assert _is_public_read("/api/health")
    assert _is_public_read("/api/readyz")


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


# ---------------------------------------------------------------------------------------------------
# The mutating half of the route table.
#
# Everything above enumerates GET/HEAD only. No mutating route was ever inspected, which is how a
# query-string credential exception scoped to the /api/events/ prefix - covering PATCH and DELETE
# /api/events/{id} - stayed invisible until somebody read the middleware by hand. The startup backstop
# declines to check writes too, on the reasoning that a write cannot reach a route without clearing the
# token gate. True, and it says nothing about WHICH role, which is the part that was wrong.
# ---------------------------------------------------------------------------------------------------

_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _api_write_routes() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for route in app.routes:
        methods = (getattr(route, "methods", None) or set()) & _WRITE_METHODS
        path = getattr(route, "path", "")
        if path.startswith("/api/") and methods:
            out.extend((m, path) for m in sorted(methods))
    return out


# Writes that must sit above the annotator floor, by what they do rather than by which router they landed
# in. Each of these changes something every other user's work is measured against, so an annotator holding
# a token should not be able to reach them.
_MUST_BE_PRIVILEGED = (
    "/api/quality/gold/seal",          # fixes the ground truth every eval and export certificate scores on
    "/api/quality/calibrate/fit",      # changes what every confidence in the system means
    "/api/activelearn/loop/retrain",   # queues GPU training; force=true bypasses the signal threshold
    "/api/hardening/slo",              # the SLO ledger the operations board reads
    "/api/govern/promote",             # which model serves
    "/api/users",                      # account administration
)


def test_the_write_scan_is_not_trivially_green():
    # The non-triviality floor, as above: a matrix over an empty route list asserts nothing.
    writes = _api_write_routes()
    assert len(writes) > 200, f"only {len(writes)} mutating /api routes found; the enumerator has broken"


def test_no_write_route_is_reachable_without_a_role():
    # _required_role never returns an anonymous floor, so a write always needs a token. Asserted rather
    # than assumed, because the whole matrix rests on it.
    anonymous = [(m, p) for m, p in _api_write_routes()
                 if _required_role(p) not in ("annotator", "reviewer", "admin")]
    assert anonymous == [], f"write routes with no role floor: {anonymous}"


def test_no_write_route_is_a_public_read():
    # A path that is both a public read and a write would be reachable unauthenticated for the write, since
    # the public-read short circuit is checked before the role floor.
    leaked = [(m, p) for m, p in _api_write_routes() if _is_public_read(p)]
    assert leaked == [], f"write routes under a public-read prefix: {leaked}"


def test_the_privileged_writes_are_above_the_annotator_floor():
    """The specific routes that were not, and the reason this matrix exists.

    Gold sealing, calibration fitting, forced retrain and the SLO ledger all sat at the annotator floor -
    /api/quality, /api/activelearn and /api/hardening are none of them reviewer prefixes - so any signed-in
    user could reseal the ground truth, refit the confidence curve, occupy the GPU, or write the
    observability board. They carry their own require_role now; adding the prefixes instead would have
    gated the READS as well, because the floor applies to any request that is not a public read.
    """
    from services.api.deps import (
        require_role,  # noqa: F401  (imported to assert the mechanism exists)
    )

    offenders = []
    for path in _MUST_BE_PRIVILEGED:
        routes = [r for r in app.routes if getattr(r, "path", "") == path
                  and (getattr(r, "methods", None) or set()) & _WRITE_METHODS]
        assert routes, f"{path} is not mounted as a write route; this table has gone stale"
        floor_ok = _required_role(path) in ("reviewer", "admin")
        dep_ok = any("require_role" in repr(getattr(d, "call", None))
                     for r in routes for d in (getattr(getattr(r, "dependant", None), "dependencies", []) or []))
        if not (floor_ok or dep_ok):
            offenders.append(path)
    assert offenders == [], (
        f"these writes are reachable by any signed-in annotator: {offenders}")
