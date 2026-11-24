"""Generate the web app's API types from the API's own OpenAPI schema.

    python -m scripts.generate_ts_types --out web/lib/api.generated.ts

`web/lib/types.ts` and `web/lib/api.ts` describe the API by hand: fourteen hundred lines of types somebody
wrote from memory of what a route returned, with comments like "absent on rows from an older API" marking
the places where memory and server had already parted. scripts/generate_sdk.py fixed the same problem for
the Python client by deriving it from the schema; this does it for TypeScript.

What comes out:
  * one exported type per schema the server declares (every request body, and every response a route
    declares a model for);
  * `Operations`, a map from "METHOD /path" to that operation's request body, query parameters and
    response type, so a caller can write `Operations["POST /api/imports/start"]["body"]`.

A response the server has not declared a model for is typed `unknown`, not `any`. That is the honest type
for it, and it is what makes an undeclared response visible at the call site instead of silently trusted.

Deliberately a small generator in this repository's own idiom rather than an npm tool: the schema pydantic
emits uses a narrow slice of JSON Schema, the output is one readable file, and the build gains no
dependency. tests/test_generated_ts_types.py fails when the file is stale.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEADER = """// LabeloxAV API types, generated from the server's OpenAPI schema. Do not edit by hand.
//
// Regenerate with:  python -m scripts.generate_ts_types --out web/lib/api.generated.ts
//
// A response typed `unknown` is one the server has not declared a model for. Declare it on the route
// (a return annotation or response_model) and regenerate; do not cast it away here.

/* eslint-disable */
"""

_IDENT = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
METHODS = ("get", "post", "put", "patch", "delete")


def type_name(schema_name: str) -> str:
    """A schema name as a TypeScript identifier. Pydantic emits names like `Body_upload__post` and
    `Page[TriageRow]`; anything that is not an identifier character becomes an underscore."""
    name = re.sub(r"[^A-Za-z0-9_$]", "_", schema_name)
    return name if not name[:1].isdigit() else f"_{name}"


def _prop(name: str) -> str:
    return name if _IDENT.match(name) else json.dumps(name)


def _literal(value) -> str:
    return "null" if value is None else json.dumps(value)


def ts_type(schema: dict | bool | None, indent: int = 0) -> str:
    """The TypeScript type for one JSON Schema node, as pydantic and FastAPI emit them."""
    if schema is True or schema is None or schema == {}:
        return "unknown"
    if schema is False:
        return "never"
    if "$ref" in schema:
        return type_name(schema["$ref"].rsplit("/", 1)[-1])
    if "const" in schema:
        return _literal(schema["const"])
    if "enum" in schema:
        return " | ".join(_literal(v) for v in schema["enum"]) or "never"
    for key, joiner in (("anyOf", " | "), ("oneOf", " | "), ("allOf", " & ")):
        if key in schema:
            parts = []
            for sub in schema[key]:
                t = ts_type(sub, indent)
                # A union inside an intersection (or the reverse) must keep its grouping.
                parts.append(f"({t})" if (" | " in t or " & " in t) and len(schema[key]) > 1 else t)
            seen: list[str] = []
            for p in parts:
                if p not in seen:
                    seen.append(p)
            return joiner.join(seen) or "unknown"

    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(dict.fromkeys(ts_type({**schema, "type": k}, indent) for k in kind))
    if kind in ("integer", "number"):
        return "number"
    if kind == "string":
        return "string"
    if kind == "boolean":
        return "boolean"
    if kind == "null":
        return "null"
    if kind == "array":
        if "prefixItems" in schema:
            return "[" + ", ".join(ts_type(s, indent) for s in schema["prefixItems"]) + "]"
        item = ts_type(schema.get("items"), indent)
        return f"({item})[]" if (" | " in item or " & " in item) else f"{item}[]"
    if kind == "object" or "properties" in schema or "additionalProperties" in schema:
        return _object(schema, indent)
    return "unknown"


def _object(schema: dict, indent: int) -> str:
    props: dict = schema.get("properties") or {}
    extra = schema.get("additionalProperties")
    if not props:
        if extra is None or extra is True or extra == {}:
            return "Record<string, unknown>"
        if extra is False:
            return "Record<string, never>"
        return f"Record<string, {ts_type(extra, indent)}>"
    required = set(schema.get("required") or [])
    pad, inner = "  " * indent, "  " * (indent + 1)
    lines = ["{"]
    for name in props:                                   # schema order, which is declaration order
        sub = props[name]
        doc = (sub.get("description") or "").strip() if isinstance(sub, dict) else ""
        if doc:
            lines.append(f"{inner}/** {doc.splitlines()[0].replace('*/', '* /')} */")
        opt = "" if name in required else "?"
        lines.append(f"{inner}{_prop(name)}{opt}: {ts_type(sub, indent + 1)};")
    if extra not in (None, False):
        lines.append(f"{inner}[key: string]: unknown;")
    lines.append(pad + "}")
    return "\n".join(lines)


def _json_schema(content: dict | None) -> dict | None:
    if not content:
        return None
    body = content.get("application/json") or next(
        (v for k, v in content.items() if k.endswith("json")), None)
    return (body or {}).get("schema") if body else None


def _response_type(op: dict) -> str:
    responses = op.get("responses") or {}
    for code in ("200", "201", "202"):
        if code in responses:
            schema = _json_schema(responses[code].get("content"))
            return ts_type(schema, 2) if schema else "unknown"
    if "204" in responses:
        return "void"
    return "unknown"


def _params(op: dict, where: str) -> str | None:
    rows = [p for p in op.get("parameters") or [] if p.get("in") == where]
    if not rows:
        return None
    parts = [f"{_prop(p['name'])}{'' if p.get('required') else '?'}: {ts_type(p.get('schema'), 3)}" for p in rows]
    return "{ " + "; ".join(parts) + " }"


def generate(spec: dict) -> str:
    out = [HEADER]
    schemas: dict = (spec.get("components") or {}).get("schemas") or {}
    for name in sorted(schemas):
        schema = schemas[name]
        doc = (schema.get("description") or "").strip()
        if doc:
            first = doc.splitlines()[0].replace("*/", "* /")
            out.append(f"/** {first} */")
        out.append(f"export type {type_name(name)} = {ts_type(schema)};\n")

    out.append("/** Every operation the server exposes, by method and path. */")
    out.append("export type Operations = {")
    n_ops = n_typed = 0
    for path in sorted(spec.get("paths") or {}):
        item = spec["paths"][path]
        for method in METHODS:
            op = item.get(method)
            if not op:
                continue
            n_ops += 1
            response = _response_type(op)
            n_typed += response != "unknown"
            fields = []
            body = _json_schema((op.get("requestBody") or {}).get("content"))
            if body:
                fields.append(f"body: {ts_type(body, 2)};")
            for where in ("path", "query"):
                p = _params(op, where)
                if p:
                    fields.append(f"{where}: {p};")
            fields.append(f"response: {response};")
            out.append(f"  {json.dumps(f'{method.upper()} {path}')}: {{")
            out.extend(f"    {f}" for f in fields)
            out.append("  };")
    out.append("};\n")
    out.append("/** The body a request to this operation takes. */")
    out.append('export type Body<K extends keyof Operations> = Operations[K] extends { body: infer B } ? B : never;')
    out.append("/** What this operation returns (`unknown` until the server declares it). */")
    out.append('export type Result<K extends keyof Operations> = Operations[K]["response"];\n')
    out.append(f"export const API_SCHEMA_STATS = {{ operations: {n_ops}, typedResponses: {n_typed}, "
               f"schemas: {len(schemas)} }} as const;")
    return "\n".join(out) + "\n"


def load_spec() -> dict:
    """The live application's schema, with every optional pack on so the output does not depend on which
    packs this machine happens to enable (the same rule scripts/generate_sdk.py follows)."""
    import os

    from core.config import get_settings
    from packs.registry import optional_pack_ids

    os.environ["LBX_PACKS__OPTIONAL"] = json.dumps(optional_pack_ids())
    get_settings.cache_clear()
    from services.api.main import app

    return app.openapi()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("web/lib/api.generated.ts"))
    ap.add_argument("--spec", type=Path, help="read an OpenAPI JSON file instead of importing the app")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else load_spec()
    text = generate(spec)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
