"""Label Studio JSON import.

Label Studio exports a list of tasks, each with `annotations` (human) and optionally `predictions` (model).
Only human annotations are imported by default: importing predictions as though a person had made them would
poison the gold pool and quietly corrupt every downstream precision number.

The important quirk: Label Studio stores rectangle geometry in PERCENT of image size, not pixels, and its `x`
`y` `width` `height` are the top-left plus extents. Reading those numbers as pixels produces boxes clustered in
the top-left corner of every image, which looks like a model failure rather than an import bug, so the
conversion is explicit and needs original_width/original_height (carried on each result).
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_labelstudio")


def _label_of(value: dict, keys=("rectanglelabels", "polygonlabels", "labels", "choices")) -> str | None:
    for k in keys:
        v = value.get(k)
        if isinstance(v, list) and v:
            return str(v[0])
        if isinstance(v, str) and v:
            return v
    return None


def _rect_to_bbox(value: dict, ow: float, oh: float) -> list[float] | None:
    """Label Studio rectangles are percentages of the original image size."""
    try:
        x, y = float(value["x"]), float(value["y"])
        w, h = float(value["width"]), float(value["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if ow <= 0 or oh <= 0:
        return None
    x1, y1 = x / 100.0 * ow, y / 100.0 * oh
    return [x1, y1, x1 + w / 100.0 * ow, y1 + h / 100.0 * oh]


def _poly_to_points(value: dict, ow: float, oh: float) -> list[list[float]]:
    pts = value.get("points") or []
    out = []
    for p in pts:
        try:
            out.append([float(p[0]) / 100.0 * ow, float(p[1]) / 100.0 * oh])
        except (IndexError, TypeError, ValueError):
            continue
    return out


def _image_ref(task: dict) -> str | None:
    data = task.get("data") or {}
    for k in ("image", "image_url", "url", "img", "media"):
        v = data.get(k)
        if isinstance(v, str) and v:
            return v
    # fall back to the first string value that looks like a path or uri
    for v in data.values():
        if isinstance(v, str) and ("/" in v or v.startswith("s3://")):
            return v
    return None


def _load_tasks(root: Path) -> list[dict]:
    tasks: list[dict] = []
    for p in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, list) and doc and isinstance(doc[0], dict) and (
                "annotations" in doc[0] or "predictions" in doc[0] or "data" in doc[0]):
            tasks.extend(doc)
        elif isinstance(doc, dict) and ("annotations" in doc or "data" in doc):
            tasks.append(doc)
    return tasks


def parse(root: Path, include_predictions: bool = False) -> list[ImportFrame]:
    tasks = _load_tasks(root)
    if not tasks:
        raise FileNotFoundError("no Label Studio task json found under the dataset")

    frames: list[ImportFrame] = []
    skipped_no_size = 0
    for task in tasks:
        ref = _image_ref(task)
        if not ref:
            continue
        groups = list(task.get("annotations") or [])
        if include_predictions:
            groups += list(task.get("predictions") or [])
        objs: list[ImportObject] = []
        width = height = None

        for group in groups:
            is_pred = "model_version" in group and group not in (task.get("annotations") or [])
            for res in group.get("result") or []:
                value = res.get("value") or {}
                ow = float(res.get("original_width") or 0)
                oh = float(res.get("original_height") or 0)
                if ow and oh:
                    width, height = int(ow), int(oh)
                rtype = res.get("type")
                label = _label_of(value) or "object_fallback"
                prov = {"source_tool": "label_studio", "prediction": bool(is_pred)}

                if rtype == "rectanglelabels" or (rtype == "rectangle" and value.get("width") is not None):
                    if not ow or not oh:
                        skipped_no_size += 1
                        continue
                    bbox = _rect_to_bbox(value, ow, oh)
                    if bbox:
                        objs.append(ImportObject(name=label, bbox=bbox, provenance=prov,
                                                 rot_deg=float(value.get("rotation") or 0.0)))

                elif rtype in ("polygonlabels", "polygon"):
                    if not ow or not oh:
                        skipped_no_size += 1
                        continue
                    pts = _poly_to_points(value, ow, oh)
                    if len(pts) >= 3:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        objs.append(ImportObject(
                            name=label, bbox=[min(xs), min(ys), max(xs), max(ys)],
                            mask_polygons=[[c for p in pts for c in p]], provenance=prov))

                elif rtype == "choices":
                    # a whole-image classification: no geometry, so it becomes a frame-level tag on every
                    # object rather than a phantom box
                    continue

        if objs:
            frames.append(ImportFrame(image_ref=ref, width=width, height=height, objects=objs))

    if skipped_no_size:
        # Loud, because the alternative is boxes silently piled in the top-left of every image.
        log.warning("import_labelstudio.skipped_without_original_size", n=skipped_no_size,
                    detail="percent geometry cannot be converted to pixels without original_width/height")
    log.info("import_labelstudio.parsed", frames=len(frames), tasks=len(tasks))
    return frames
