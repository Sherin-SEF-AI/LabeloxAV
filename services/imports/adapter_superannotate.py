"""SuperAnnotate export import.

SuperAnnotate writes one JSON per image, named `<image>.json`, alongside a `classes/classes.json` holding the
project's taxonomy. Boxes arrive as `points: {x1, y1, x2, y2}` in pixels, which is already the shape this
system uses, so the geometry needs no conversion at all.

The class name lives in `className` on newer exports and only as `classId` on older ones, which is why the
taxonomy file is read rather than ignored: an export whose annotations carry ids and whose ids are never
resolved imports every object as "unknown", and a migration that arrives with one class is one nobody
notices is broken until they try to train on it.

Attributes are a list of `{name, groupName}` rather than a map, because SuperAnnotate models them as
selections within named groups. They are folded into a map keyed by group so the shape matches every other
adapter, and the group name is kept, since "colour: red" and "state: red" are different facts.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_superannotate")


def _taxonomy(root: Path) -> dict[int, str]:
    """classId -> name, from classes/classes.json when it is present."""
    out: dict[int, str] = {}
    for p in list(root.rglob("classes.json")) + list(root.rglob("classes/*.json")):
        try:
            doc = json.loads(p.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        for c in (doc if isinstance(doc, list) else doc.get("classes") or []):
            cid, name = c.get("id"), c.get("name")
            if cid is not None and name:
                out[int(cid)] = str(name)
    return out


def _attrs(instance: dict) -> dict:
    """SuperAnnotate attributes are selections inside named groups, not a flat map."""
    out: dict = {}
    for a in instance.get("attributes") or []:
        name, group = a.get("name"), a.get("groupName")
        if not name:
            continue
        # Group kept, because "colour: red" and "signal state: red" are different facts and flattening them
        # to "red" loses the one that matters.
        out[str(group or "attribute")] = str(name)
    return out


def parse(root: Path) -> list[ImportFrame]:
    taxonomy = _taxonomy(root)
    frames: list[ImportFrame] = []
    skipped = unresolved = 0

    for p in sorted(root.rglob("*.json")):
        if p.name == "classes.json" or "classes" in p.parts:
            continue
        try:
            doc = json.loads(p.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata") or {}
        image_ref = meta.get("name") or p.stem
        fr = ImportFrame(image_ref=str(image_ref),
                         width=meta.get("width"), height=meta.get("height"))

        for inst in doc.get("instances") or []:
            pts = inst.get("points") or {}
            try:
                bbox = [float(pts["x1"]), float(pts["y1"]), float(pts["x2"]), float(pts["y2"])]
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            name = inst.get("className")
            if not name and inst.get("classId") is not None:
                name = taxonomy.get(int(inst["classId"]))
            if not name:
                unresolved += 1
                name = "unknown"
            fr.objects.append(ImportObject(
                name=str(name), bbox=bbox, attrs=_attrs(inst),
                track_ref=(str(inst["trackingId"]) if inst.get("trackingId") else None),
            ))
        if fr.objects or fr.width:
            frames.append(fr)

    log.info("import.superannotate.parsed", frames=len(frames),
             objects=sum(len(f.objects) for f in frames), taxonomy=len(taxonomy),
             skipped=skipped, unresolved_class_ids=unresolved)
    return frames
