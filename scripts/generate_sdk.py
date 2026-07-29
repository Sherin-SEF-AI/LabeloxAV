"""Generate the Python SDK from the API's own OpenAPI schema.

`sdk/labelox_client.py` was written by hand, which means it describes what somebody remembered the API did
at the moment they wrote it. The API now has over five hundred routes; the hand-rolled client covers a few
dozen, its parameter names drift from the server's the moment a route changes, and nothing detects either.

Generating from the schema fixes the direction of truth: the server defines the surface, the client is
derived, and a route that changes shape produces a client that changes shape rather than one that keeps
calling the old one.

Deliberately not openapi-generator or datamodel-code-generator. Both would add a Java or a heavy Python
toolchain to the build for a client that needs one file, and both produce a package nobody reads. This
emits a single module in the same idiom as the rest of the codebase, with the route table visible in it.

    python -m scripts.generate_sdk --out sdk/generated_client.py
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
from pathlib import Path

HEADER = '''"""LabeloxAV Python client, generated from the API's OpenAPI schema.

Do not edit by hand. Regenerate with:

    python -m scripts.generate_sdk --out sdk/generated_client.py

Every method here corresponds to one route on the server, with its real path, method and parameters. The
hand-written client in `labelox_client.py` remains for the ergonomic helpers that compose several calls;
this is the complete, always-current surface underneath it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class LabeloxError(RuntimeError):
    """An API call failed. Carries the status and the server's own message."""

    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        super().__init__(f"{method} {path} -> {status}: {detail}")
        self.status, self.method, self.path, self.detail = status, method, path, detail


class LabeloxClient:
    """A thin, complete client over the REST API.

    The token is required rather than optional. Reads are deny-by-default on the server, so a client
    constructed without one fails on its first call with a 401 that looks like a server problem; refusing
    at construction says what is actually wrong.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.environ.get("LABELOX_URL")
                         or "http://localhost:8000").rstrip("/")
        self.token = token or os.environ.get("LABELOX_TOKEN")
        if not self.token:
            raise LabeloxError(0, "INIT", "", (
                "no token. Pass token= or set LABELOX_TOKEN; reads are deny-by-default on the server, so "
                "an unauthenticated client fails on its first call with a 401 that looks like an outage."))
        self._client = httpx.Client(timeout=timeout,
                                    headers={"Authorization": f"Bearer {self.token}"})

    def __enter__(self) -> LabeloxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, *, params: dict | None = None,
              json_body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.request(method, url, params=_clean(params), json=json_body)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:  # noqa: BLE001
                detail = r.text[:400]
            raise LabeloxError(r.status_code, method, path, str(detail))
        if not r.content:
            return None
        try:
            return r.json()
        except Exception:  # noqa: BLE001 - a binary body (a crop, an export) is returned as bytes
            return r.content

    # ---- generated methods ----
'''

FOOTER = '''

def _clean(params: dict | None) -> dict | None:
    """Drop unset query parameters.

    Sending them as the string "None" is the classic generated-client bug: the server sees a value where
    the caller meant absence, and a filter nobody asked for silently applies.
    """
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}
'''

# Routes a generated client has no business exposing. Each is either a browser-only flow or a credential
# path where a scripted caller wants the explicit method rather than a generated one.
SKIP_PREFIXES = ("/api/auth/oidc", "/api/auth/dev-login", "/api/events/")


def _method_name(method: str, path: str, seen: set[str]) -> str:
    """A readable Python name for a route, unique within the client."""
    parts = [p for p in path.replace("/api/", "").split("/") if p]
    words: list[str] = []
    for p in parts:
        if p.startswith("{"):
            words.append("by_" + re.sub(r"[^a-z0-9]+", "_", p.strip("{}").lower()))
        else:
            words.append(re.sub(r"[^a-z0-9]+", "_", p.lower()))
    verb = {"get": "get", "post": "post", "put": "put", "patch": "patch",
            "delete": "delete"}.get(method.lower(), method.lower())
    base = "_".join([verb, *words]).strip("_")
    base = re.sub(r"_+", "_", base)
    if keyword.iskeyword(base):
        base += "_"
    name, n = base, 2
    while name in seen:
        name, n = f"{base}_{n}", n + 1
    seen.add(name)
    return name


def _py_type(schema: dict | None) -> str:
    if not schema:
        return "Any"
    t = schema.get("type")
    return {"string": "str", "integer": "int", "number": "float",
            "boolean": "bool", "array": "list", "object": "dict"}.get(t, "Any")


def _safe_param(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if keyword.iskeyword(s) or s in ("self", "params", "json_body"):
        s += "_"
    if s and s[0].isdigit():
        s = f"p_{s}"
    return s


def generate(schema: dict) -> str:
    """Emit the client module from a parsed OpenAPI document."""
    out: list[str] = [HEADER]
    seen: set[str] = set()
    count = 0

    for path, ops in sorted((schema.get("paths") or {}).items()):
        if path.startswith(SKIP_PREFIXES):
            continue
        for method, op in sorted(ops.items()):
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            name = _method_name(method, path, seen)
            summary = (op.get("summary") or op.get("description") or "").strip().split("\n")[0]

            path_params, query_params = [], []
            for p in op.get("parameters") or []:
                target = path_params if p.get("in") == "path" else query_params
                if p.get("in") in ("path", "query"):
                    target.append(p)

            # Required arguments first, defaulted ones after, whatever order the schema lists them in.
            # Python forbids the reverse, and a generator that emits it produces a module that does not
            # import at all rather than one method that misbehaves.
            required: list[str] = []
            optional: list[str] = []
            for p in path_params:
                required.append(f"{_safe_param(p['name'])}: {_py_type(p.get('schema'))}")
            for p in query_params:
                py = _safe_param(p["name"])
                typ = _py_type(p.get("schema"))
                if p.get("required"):
                    required.append(f"{py}: {typ}")
                else:
                    default = p.get("schema", {}).get("default")
                    optional.append(f"{py}: {typ} | None = {default!r}"
                                    if default is not None else f"{py}: {typ} | None = None")
            has_body = bool(op.get("requestBody"))
            # The body sits at the head of the defaulted group, so `client.post_x(id, {...})` reads the
            # way a caller expects rather than requiring every optional filter to be named first.
            body_arg = ["body: Any = None"] if has_body else []
            args: list[str] = ["self", *required, *body_arg, *optional]

            # The path is formatted from its own parameters, so a caller cannot transpose two ids and get
            # a request that succeeds against the wrong resource.
            fmt = path
            for p in path_params:
                fmt = fmt.replace("{" + p["name"] + "}", "{" + _safe_param(p["name"]) + "}")

            query = ("{" + ", ".join(f'"{p["name"]}": {_safe_param(p["name"])}'
                                     for p in query_params) + "}") if query_params else "None"

            out.append(f"    def {name}({', '.join(args)}) -> Any:")
            if summary:
                out.append(f'        """{summary}"""')
            out.append(f'        return self._call("{method.upper()}", f"{fmt}",')
            out.append(f"                          params={query},"
                       f" json_body={'body' if has_body else 'None'})")
            out.append("")
            count += 1

    out.append(FOOTER)
    out.append(f"\n# {count} routes generated from the server schema.\n")
    return "\n".join(out)


def load_schema(source: str) -> dict:
    """Read the schema from a file, a URL, or the app itself.

    The in-process path is the default because it needs no running server: a build step that required the
    API to be up would generate a stale client whenever it was not.
    """
    if source == "app":
        from services.api.main import app

        return app.openapi()
    if source.startswith("http"):
        import httpx

        return httpx.get(source, timeout=30).json()
    return json.loads(Path(source).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", default="app",
                    help="'app' (in-process, the default), a URL, or a path to openapi.json")
    ap.add_argument("--out", default="sdk/generated_client.py")
    ap.add_argument("--check", action="store_true",
                    help="fail if the generated client differs from the file on disk")
    args = ap.parse_args()

    code = generate(load_schema(args.schema))
    dest = Path(args.out)

    if args.check:
        current = dest.read_text() if dest.exists() else ""
        if current.strip() != code.strip():
            raise SystemExit(
                f"{dest} is out of date with the API schema. Regenerate with "
                f"`python -m scripts.generate_sdk --out {dest}`.")
        print(f"{dest} is up to date")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(code)
    print(f"wrote {dest} ({code.count('    def ')} methods)")


if __name__ == "__main__":
    main()
