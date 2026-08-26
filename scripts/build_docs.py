"""Generate the parts of the documentation site that are derived from the code.

Two things must never be hand-maintained, because a hand-maintained copy of either is a copy that goes
quietly stale and is then quoted at somebody: the OpenAPI schema, and the route inventory.

    .venv/bin/python -m scripts.build_docs

Writes `docs/api/openapi.json` and `docs/api/inventory.md`. The site build runs this first, so a route added
without documentation still appears, and a route deleted stops appearing.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "api"


def _schema() -> dict:
    from services.api.main import app

    return app.openapi()


def _inventory(schema: dict) -> str:
    """Every route, grouped by tag, with its auth floor where the router declares one."""
    by_tag: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for path, item in sorted(schema.get("paths", {}).items()):
        for method, op in item.items():
            if not isinstance(op, dict) or method.upper() not in (
                    "GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            tag = (op.get("tags") or ["untagged"])[0]
            summary = (op.get("summary") or "").strip()
            by_tag[tag].append((method.upper(), path, summary))

    total = sum(len(v) for v in by_tag.values())
    lines = [
        "# Route inventory",
        "",
        f"{total} routes across {len(by_tag)} tag groups, generated from the running FastAPI app by",
        "`scripts/build_docs.py`. The interactive reference is [REST API](rest.md).",
        "",
        "!!! warning \"Response schemas are thin\"",
        "    Only a handful of routes declare a `response_model`, so most entries below document the",
        "    request side and the status codes, not the response body. That is a real gap, recorded here",
        "    rather than papered over: the generated SDK is correspondingly untyped.",
        "",
    ]
    for tag in sorted(by_tag):
        rows = sorted(by_tag[tag], key=lambda r: (r[1], r[0]))
        lines += [f"## {tag}", "", f"{len(rows)} routes.", "",
                  "| Method | Path | Summary |", "| --- | --- | --- |"]
        for method, path, summary in rows:
            lines.append(f"| `{method}` | `{path}` | {summary or '-'} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    (OUT / "openapi.json").write_text(json.dumps(schema, indent=2, sort_keys=True))
    (OUT / "inventory.md").write_text(_inventory(schema))
    n_paths = len(schema.get("paths", {}))
    n_schemas = len(schema.get("components", {}).get("schemas", {}))
    print(f"wrote docs/api/openapi.json ({n_paths} paths, {n_schemas} schemas) and docs/api/inventory.md")


if __name__ == "__main__":
    main()
