"""Encord export import.

Encord exports a label row per data unit, with `data_units` keyed by hash and each unit's `labels` keyed by
frame number, so a video's frames arrive as numbered entries under one unit rather than as separate files.
That structure is the whole reason this adapter exists separately: flattening it wrongly gives every frame
the same image reference and silently collapses a whole clip onto one frame.

Geometry is normalised: `boundingBox: {x, y, w, h}` with every value in 0..1 of the image size. Reading those
as pixels puts every box in the top-left corner, which looks like a catastrophic model failure rather than an
import bug, so the conversion is explicit and needs the unit's width and height.

Class names come from the ontology the customer built in Encord, carried on each object as `name`. They are
returned untranslated for the same reason as every other adapter: mapping happens once, in
`services/imports/remap.py`, so a migration's taxonomy conflicts surface in one report.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_encord")


def _docs(root: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(p.read_text(errors="ignore"))
        except json.JSONDecodeError:
            continue
        out.extend(doc if isinstance(doc, list) else [doc])
    return out


def _bbox(obj: dict, w: float, h: float) -> list[float] | None:
    """Encord bounding boxes are normalised to 0..1 of the image."""
    bb = obj.get("boundingBox") or obj.get("bounding_box")
    if not bb:
        return None
    try:
        x, y = float(bb["x"]) * w, float(bb["y"]) * h
        bw, bh = float(bb["w"]) * w, float(bb["h"]) * h
    except (KeyError, TypeError, ValueError):
        return None
    if bw <= 0 or bh <= 0:
        return None
    return [x, y, x + bw, y + bh]


def parse(root: Path) -> list[ImportFrame]:
    frames: list[ImportFrame] = []
    skipped = no_size = 0

    for doc in _docs(root):
        for unit in (doc.get("data_units") or {}).values():
            w, h = unit.get("width"), unit.get("height")
            if not w or not h:
                # Without the unit's size the normalised geometry cannot be converted, and guessing a size
                # would place every box somewhere plausible and wrong.
                no_size += 1
                continue
            title = unit.get("data_title") or unit.get("data_hash") or "unknown"
            labels = unit.get("labels") or {}

            # A still image carries its objects directly; a video carries them under frame numbers. Treating
            # the second as the first gives every frame of a clip the same reference and collapses it.
            per_frame = (labels if any(k.isdigit() for k in labels) else {"0": labels})

            for frame_no, entry in per_frame.items():
                objs = (entry or {}).get("objects") or []
                if not objs:
                    continue
                # A numbered frame gets a reference that names it, so two frames of one clip stay distinct.
                ref = f"{title}#{frame_no}" if any(k.isdigit() for k in labels) else str(title)
                fr = ImportFrame(image_ref=ref, width=int(w), height=int(h))
                for o in objs:
                    bbox = _bbox(o, float(w), float(h))
                    if bbox is None:
                        skipped += 1
                        continue
                    fr.objects.append(ImportObject(
                        name=str(o.get("name") or o.get("value") or "unknown"),
                        bbox=bbox,
                        # Encord's objectHash is stable across frames of a clip, which is exactly a track.
                        track_ref=(str(o["objectHash"]) if o.get("objectHash") else None),
                        attrs={c.get("name", "attr"): c.get("answers")
                               for c in (o.get("classifications") or []) if c.get("answers") is not None},
                    ))
                if fr.objects:
                    frames.append(fr)

    log.info("import.encord.parsed", frames=len(frames),
             objects=sum(len(f.objects) for f in frames), skipped=skipped, units_without_size=no_size)
    return frames
