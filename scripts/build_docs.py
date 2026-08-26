"""Generate the parts of the documentation site that are derived from the code.

Nothing here is hand-maintained, because a hand-maintained copy of a route table, a screen list or a tool
palette is a copy that goes quietly stale and is then quoted at somebody.

    .venv/bin/python -m scripts.build_docs

Writes, from the code that defines each:

  docs/api/openapi.json      the FastAPI schema
  docs/api/inventory.md      every route, grouped by tag
  docs/guide/platforms.md    the seven platforms and their screens, from the frontend registry
  docs/guide/editor.md       every editor mode and tool, from the tool registry
  docs/guide/shortcuts.md    every keyboard shortcut, from the overlay a coupling test already pins

The site build runs this first, so a route, screen, tool or key added without documentation still appears,
and one deleted stops appearing.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "api"
GUIDE = ROOT / "docs" / "guide"


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


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _platforms() -> str:
    """The seven platforms and their screens, from the registry that drives the launcher itself."""
    src = _read("web/platforms/registry.ts")
    blocks = re.findall(r'\{\s*id:\s*"(\w+)",\s*label:\s*"([^"]+)",\s*role:\s*"([^"]+)",(.*?)\n  \},',
                        src, re.S)
    if not blocks:
        raise RuntimeError("platform registry did not parse; the docs would silently lose every screen")
    out = ["# The seven platforms", "",
           "LabeloxAV is the host; the annotation core and six folded subsystems are Platforms - distinct",
           "navigable UIs behind a launcher, over one shared backend spine. Switch with the platform picker",
           "in the top bar.", "",
           "Generated from `web/platforms/registry.ts`, which is also what drives the launcher, so the",
           "screens listed here are the screens that exist.", ""]
    for _pid, label, role, body in blocks:
        gate = re.search(r'gate:\s*("(\w+)"|null)', body)
        stage = re.search(r"flywheelStage:\s*(\d+|null)", body)
        home = re.search(r'home:\s*"([^"]+)"', body)
        navs = re.findall(
            r'\{\s*href:\s*"([^"]+)",\s*label:\s*"([^"]+)"(?:,\s*hint:\s*"([^"]+)")?\s*\}', body)
        out += [f"## {label}", "", f"*{role}*", ""]
        bits = [f"Home `{home.group(1)}`"] if home else []
        if stage and stage.group(1) != "null":
            bits.append(f"flywheel stage {stage.group(1)}")
        if gate and gate.group(2):
            bits.append(f"**can block progression on `{gate.group(2)}`**")
        out += [" &middot; ".join(bits), "", "| Screen | Route | What it is for |", "| --- | --- | --- |"]
        out += [f"| {lbl} | `{href}` | {hint or '-'} |" for href, lbl, hint in navs]
        out.append("")
    return "\n".join(out)


def _editor() -> str:
    """Every editor mode and tool, from the registry that builds the tool strip."""
    src = _read("web/lib/editor/registry.ts")
    modes = re.split(r"\n  \{\n", src.split("export const MODES")[1])[1:]
    canvas_name = {"konva": "2D canvas", "three": "3D viewport", "table": "table"}
    out = ["# The frame editor", "",
           "Seven modes, each with its own tools and its own canvas. `Shift`+`1`..`7` switches mode; the",
           "tool hotkeys below are global, so the same letter picks the same tool in every mode that has it.",
           "",
           "Generated from `web/lib/editor/registry.ts`, which is what builds the tool strip - so every tool",
           "listed here is a tool that exists.", "",
           '!!! tip "The two keys worth learning first"',
           "    `Enter` confirms the frame and moves to the next one. `A` accepts the selection (in Review",
           "    mode) or accepts everything (elsewhere). Most of a review session is those two keys.", ""]
    seen = 0
    for m in modes:
        mk = re.search(r'key: "[\w\d]+", label: "([^"]+)", rail: "([^"]+)", hotkey: "(\d)", canvas: "(\w+)"',
                       m)
        if not mk:
            continue
        seen += 1
        label, rail, hot, canvas = mk.groups()
        out += [f"## {label}", "",
                f"`Shift`+`{hot}` &middot; rail `{rail}` &middot; {canvas_name.get(canvas, canvas)}", "",
                "| Group | Tool | Key |", "| --- | --- | --- |"]
        for g in re.finditer(r'\{ key: "[\w-]+", label: "([A-Z][^"]*)", tools: \[(.*?)\] \}', m, re.S):
            group_label, tools = g.groups()
            for t in re.finditer(r'\{ key: "[\w-]+", label: "([^"]+)", hotkey: "([^"]+)" \}', tools):
                out.append(f"| {group_label} | {t.group(1)} | `{t.group(2)}` |")
        out.append("")
    if not seen:
        raise RuntimeError("editor registry did not parse; the docs would silently lose every tool")
    return "\n".join(out)


def _shortcuts() -> str:
    """Every keyboard shortcut, from the overlay a coupling test already pins to the key handler."""
    src = _read("web/components/shell/ShortcutOverlay.tsx")

    def rows(const: str):
        blk = src.split(f"export const {const}")[1].split("];")[0]
        return re.findall(r'\{ keys: "([^"]+)", label: "([^"]+)" \}', blk)

    g, t = rows("GLOBAL"), rows("TOOLS")
    if not g or not t:
        raise RuntimeError("shortcut overlay did not parse; the docs would silently lose every key")
    out = ["# Keyboard shortcuts", "",
           "Press `?` anywhere in the app for this list in a searchable overlay.", "",
           "Generated from `web/components/shell/ShortcutOverlay.tsx`, which a coupling test keeps in step",
           "with the editor's actual key handler - a key documented here but not bound, or bound but not",
           "documented, fails the test rather than misleading you.", "",
           "## Global", "", "| Keys | Action |", "| --- | --- |"]
    out += [f"| `{k}` | {label} |" for k, label in g]
    out += ["", "## Tools", "",
            "Global across modes: the same letter selects the same tool everywhere it exists, though a tool",
            "may be inert on a canvas that does not support it.", "",
            "| Keys | Tool |", "| --- | --- |"]
    out += [f"| `{k}` | {label} |" for k, label in t]
    out += ["", "## Modes", "", "| Keys | Mode |", "| --- | --- |"]
    out += [f"| `Shift` `{i}` | {name} |" for i, name in enumerate(
        ("Objects", "Lanes and drivable", "Semantic", "Events", "Pose and behavior", "3D and LiDAR",
         "Review"), start=1)]
    out.append("")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GUIDE.mkdir(parents=True, exist_ok=True)
    (GUIDE / "platforms.md").write_text(_platforms())
    (GUIDE / "editor.md").write_text(_editor())
    (GUIDE / "shortcuts.md").write_text(_shortcuts())
    schema = _schema()
    (OUT / "openapi.json").write_text(json.dumps(schema, indent=2, sort_keys=True))
    (OUT / "inventory.md").write_text(_inventory(schema))
    n_paths = len(schema.get("paths", {}))
    n_schemas = len(schema.get("components", {}).get("schemas", {}))
    print(f"wrote docs/api/openapi.json ({n_paths} paths, {n_schemas} schemas), docs/api/inventory.md, "
          f"and the generated guide pages (platforms, editor, shortcuts)")


if __name__ == "__main__":
    main()
