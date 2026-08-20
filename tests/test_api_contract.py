"""The API's own description of itself, and a ratchet on how much of it is typed.

An untyped route returns `dict`. That is fine for the handler and expensive everywhere else: the generated
OpenAPI cannot say what comes back, so `scripts/generate_sdk.py` produces a client whose methods all return
`Any`, and nothing catches a response shape changing under a caller. 41 of 673 routes declare a
`response_model` today, which is about 6%.

This does not demand they all be typed at once - that would be a thousand-line change nobody could review,
and a gate that starts red gets switched off. It pins the number so it can only go up, in the same shape as
the mypy strict allowlist and the swallowed-failure baseline.

Pure: it introspects the mounted app, no DB.
"""

from __future__ import annotations

from services.api.main import app

# Measured 2026-08-18. Raise it when you type more routes; it must never be lowered to make a change pass.
MIN_TYPED_ROUTES = 41

# Non-triviality floors. A ratchet over an empty route table is a statement about nothing.
MIN_API_ROUTES = 500
MIN_SCHEMA_PATHS = 500


def _api_routes():
    return [r for r in app.routes if getattr(r, "path", "").startswith("/api/")]


def test_the_route_table_is_not_empty():
    routes = _api_routes()
    assert len(routes) >= MIN_API_ROUTES, (
        f"only {len(routes)} /api routes mounted; the introspection has broken and every assertion below "
        "is about the empty set")


def test_the_schema_generates():
    """A schema that raises is an SDK that cannot be regenerated and a client nobody can write.

    FastAPI builds this lazily, so a route with an un-serialisable annotation does not fail at import or at
    request time - it fails the first time somebody asks for the schema, which is usually in CI on an
    unrelated PR, or never.
    """
    schema = app.openapi()
    assert schema.get("openapi"), "no openapi version in the generated schema"
    assert len(schema.get("paths", {})) >= MIN_SCHEMA_PATHS


def test_the_typed_route_count_only_grows():
    typed = [r for r in _api_routes() if getattr(r, "response_model", None) is not None]
    assert len(typed) >= MIN_TYPED_ROUTES, (
        f"{len(typed)} routes declare a response_model, down from the recorded {MIN_TYPED_ROUTES}. "
        "An untyped route makes the generated SDK return Any and lets a response shape change under every "
        "caller unnoticed. If a route legitimately lost its model, lower the floor deliberately and say why.")


def test_every_path_has_at_least_one_method():
    # A route mounted with no methods is unreachable and silently so.
    empty = [getattr(r, "path", "") for r in _api_routes() if not (getattr(r, "methods", None) or set())]
    assert empty == [], f"routes mounted with no HTTP method: {empty}"


def test_no_path_is_mounted_twice_for_the_same_method():
    """Two handlers on one (method, path) means the second is dead and nothing says so.

    FastAPI resolves the first match, so the later registration is unreachable code that looks live - and
    the pair is easy to create by mounting the same router twice under different prefixes.
    """
    seen: dict[tuple[str, str], int] = {}
    for r in _api_routes():
        for m in (getattr(r, "methods", None) or set()):
            key = (m, getattr(r, "path", ""))
            seen[key] = seen.get(key, 0) + 1
    dupes = sorted(f"{m} {p}" for (m, p), n in seen.items() if n > 1)
    assert dupes == [], f"these are registered more than once; only the first is reachable: {dupes}"


def test_operation_ids_are_unique_enough_to_generate_a_client():
    """Two operations with one id makes a generated client silently lose a method.

    FastAPI derives the id from the function name plus the path, so a collision means two handlers named
    the same thing on paths that normalise together - and the SDK generator writes one over the other.
    """
    schema = app.openapi()
    ids: dict[str, int] = {}
    for _path, ops in schema.get("paths", {}).items():
        for _method, op in ops.items():
            if not isinstance(op, dict):
                continue
            oid = op.get("operationId")
            if oid:
                ids[oid] = ids.get(oid, 0) + 1
    dupes = sorted(k for k, n in ids.items() if n > 1)
    assert dupes == [], f"duplicate operationIds; a generated client would lose a method: {dupes}"
