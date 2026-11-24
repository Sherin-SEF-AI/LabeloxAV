"""The web app's API types are derived from the server, and stay derived.

`web/lib/types.ts` is hand-written and has drifted before. The generated file is the server's own
description of itself; these check the generator's mapping from JSON Schema to TypeScript on the shapes
pydantic actually emits, and fail when the committed file no longer matches the running schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_ts_types import generate, ts_type, type_name

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("schema,expected", [
    ({"type": "string"}, "string"),
    ({"type": "integer"}, "number"),
    ({"type": "number"}, "number"),
    ({"type": "boolean"}, "boolean"),
    ({"type": "null"}, "null"),
    ({}, "unknown"),
    (True, "unknown"),
    (None, "unknown"),
    ({"$ref": "#/components/schemas/TriageRow"}, "TriageRow"),
    ({"type": "array", "items": {"type": "string"}}, "string[]"),
    ({"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "(string | null)[]"),
    ({"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}]}, "[number, number]"),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "string | null"),
    ({"anyOf": [{"type": "integer"}, {"type": "number"}]}, "number"),
    ({"enum": ["review", "accepted"]}, '"review" | "accepted"'),
    ({"const": "pending"}, '"pending"'),
    ({"type": ["string", "null"]}, "string | null"),
    ({"type": "object"}, "Record<string, unknown>"),
    ({"type": "object", "additionalProperties": {"type": "number"}}, "Record<string, number>"),
    ({"type": "object", "additionalProperties": False}, "Record<string, never>"),
])
def test_json_schema_maps_to_the_typescript_a_person_would_write(schema, expected):
    assert ts_type(schema) == expected


def test_an_object_marks_what_is_required_and_what_is_not():
    out = ts_type({"type": "object", "required": ["job_id"],
                   "properties": {"job_id": {"type": "string"},
                                  "queue_position": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}})
    assert "job_id: string;" in out
    assert "queue_position?: number | null;" in out


def test_a_property_that_is_not_an_identifier_is_quoted():
    out = ts_type({"type": "object", "properties": {"x-request-id": {"type": "string"}}})
    assert '"x-request-id"?: string;' in out


def test_a_union_inside_an_intersection_keeps_its_grouping():
    out = ts_type({"allOf": [{"$ref": "#/components/schemas/A"},
                             {"anyOf": [{"type": "string"}, {"type": "null"}]}]})
    assert out == "A & (string | null)"


@pytest.mark.parametrize("raw,expected", [("TriageRow", "TriageRow"), ("Page[TriageRow]", "Page_TriageRow_"),
                                          ("Body_upload__post", "Body_upload__post"), ("3dBox", "_3dBox")])
def test_schema_names_become_identifiers(raw, expected):
    assert type_name(raw) == expected


def test_an_undeclared_response_is_unknown_and_a_declared_one_is_typed():
    spec = {
        "components": {"schemas": {"Job": {"type": "object", "required": ["id"],
                                           "properties": {"id": {"type": "string"}}}}},
        "paths": {
            "/api/jobs/{job_id}": {
                "get": {"parameters": [{"name": "job_id", "in": "path", "required": True,
                                        "schema": {"type": "string"}},
                                       {"name": "limit", "in": "query", "schema": {"type": "integer"}}],
                        "responses": {"200": {"content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/Job"}}}}}},
                "delete": {"responses": {"204": {"description": "gone"}}},
            },
            "/api/legacy": {"post": {
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}},
                "responses": {"200": {"content": {"application/json": {"schema": {}}}}}}},
        },
    }
    out = generate(spec)
    assert 'export type Job = {' in out
    assert '"GET /api/jobs/{job_id}": {' in out
    assert "path: { job_id: string };" in out and "query: { limit?: number };" in out
    assert "response: Job;" in out
    assert '"DELETE /api/jobs/{job_id}": {\n    response: void;' in out
    assert '"POST /api/legacy": {\n    body: Job;\n    response: unknown;' in out
    assert "operations: 3, typedResponses: 2" in out


def test_generation_is_deterministic():
    spec = {"components": {"schemas": {"B": {"type": "string"}, "A": {"type": "integer"}}},
            "paths": {"/z": {"get": {"responses": {}}}, "/a": {"get": {"responses": {}}}}}
    first = generate(spec)
    assert first == generate(spec)
    assert first.index("export type A") < first.index("export type B")
    assert first.index('"GET /a"') < first.index('"GET /z"')


def test_the_committed_types_are_current_with_the_api_schema():
    """The same rule as the Python client: the server defines the surface and the client is derived."""
    from services.api.main import app

    expected = generate(app.openapi())
    on_disk = (REPO / "web" / "lib" / "api.generated.ts").read_text()
    assert on_disk.strip() == expected.strip(), (
        "web/lib/api.generated.ts is out of date; run `python -m scripts.generate_ts_types`")
