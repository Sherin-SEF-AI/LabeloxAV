"""Scale AI export import.

Scale returns one JSON document per task, or a JSONL of them, with the annotations under `response`. The
geometry is `{left, top, width, height}` in pixels, which is the same top-left-plus-extents idiom Label
Studio uses but without the percentage trap, so the conversion is arithmetic rather than a judgement.

Two things are worth stating because getting either wrong corrupts a migration silently.

Scale nests the useful attributes under a per-annotation `attributes` map whose keys are the customer's own
taxonomy, not ours. They are carried through verbatim rather than guessed at: an attribute this ontology has
no field for is still evidence somebody paid for, and dropping it during a migration is how a team discovers
the switch cost them data.

`label` is the class, and it is theirs. Nothing here maps it. Every adapter returns the external name and
`services/imports/remap.py` does the mapping in one place, so a migration's taxonomy conflicts surface in
one report rather than being resolved differently by each parser.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_scale")


def _tasks(root: Path) -> list[dict]:
    """Every task in the export, whether it arrived as one JSON array, many files, or JSONL."""
    out: list[dict] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".json", ".jsonl") or not p.is_file():
            continue
        text = p.read_text(errors="ignore").strip()
        if not text:
            continue
        if p.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        out.extend(doc if isinstance(doc, list) else [doc])
    return out


def _bbox(a: dict) -> list[float] | None:
    """Scale boxes are left/top/width/height in pixels."""
    try:
        x, y = float(a["left"]), float(a["top"])
        w, h = float(a["width"]), float(a["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def parse(root: Path) -> list[ImportFrame]:
    frames: dict[str, ImportFrame] = {}
    skipped = 0
    for task in _tasks(root):
        params = task.get("params") or {}
        image_ref = (task.get("attachment") or params.get("attachment")
                     or task.get("image") or task.get("attachment_url"))
        if not image_ref:
            skipped += 1
            continue
        response = task.get("response") or {}
        anns = response.get("annotations") or task.get("annotations") or []

        fr = frames.get(str(image_ref))
        if fr is None:
            fr = ImportFrame(image_ref=str(image_ref))
            frames[str(image_ref)] = fr

        for a in anns:
            bbox = _bbox(a)
            if bbox is None:
                skipped += 1
                continue
            fr.objects.append(ImportObject(
                name=str(a.get("label") or "unknown"),
                bbox=bbox,
                # Their taxonomy, carried verbatim. A migration that drops the attributes somebody paid to
                # collect is one the customer notices after switching, which is the worst moment.
                attrs={k: v for k, v in (a.get("attributes") or {}).items() if v is not None},
                track_ref=(str(a["uuid"]) if a.get("uuid") else None),
            ))
    log.info("import.scale.parsed", frames=len(frames),
             objects=sum(len(f.objects) for f in frames.values()), skipped=skipped)
    return list(frames.values())
