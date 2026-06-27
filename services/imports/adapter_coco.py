"""COCO import: inverts services.export.adapter_coco. Reads annotations.json, mapping each annotation
back to an ImportObject. Our own exports carry a `labelox` extension block (class_name, attrs, track,
provenance) which is used verbatim for a lossless round-trip; foreign COCO falls back to the category
name + [x,y,w,h] box. Image bytes come from the annotation's `uri` (our export) or a local image file.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports._util import find_file, load_json
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_coco")


def parse(root: Path) -> list[ImportFrame]:
    path = find_file(root, "annotations.json", "*.json")
    if path is None:
        raise FileNotFoundError("no COCO annotations.json found under the dataset")
    doc = load_json(path)
    if not isinstance(doc, dict) or "images" not in doc or "annotations" not in doc:
        raise ValueError(f"{path.name} is not a COCO file (missing images/annotations)")

    cat_name = {c["id"]: c["name"] for c in doc.get("categories", [])}
    images = {img["id"]: img for img in doc["images"]}
    by_image: dict[int, list[dict]] = {}
    for ann in doc["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    frames: list[ImportFrame] = []
    for img_id, img in images.items():
        objs: list[ImportObject] = []
        for ann in by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            bbox = [x, y, x + w, y + h]
            ext = ann.get("labelox") or {}
            name = ext.get("class_name") or cat_name.get(ann.get("category_id"), "object_fallback")
            objs.append(ImportObject(
                name=name, bbox=bbox, attrs=ext.get("attributes", {}),
                track_ref=ext.get("track_id"), conf=float(ext.get("conf", 1.0)),
                provenance=ext.get("provenance", {}),
            ))
        ref = img.get("uri") or img.get("file_name")
        if not ref:
            continue
        frames.append(ImportFrame(
            image_ref=ref, width=img.get("width"), height=img.get("height"),
            ts_ns=img.get("ts_ns"), cam_id=img.get("cam_id", "cam_front"), objects=objs,
        ))
    log.info("import_coco.parsed", frames=len(frames), file=path.name)
    return frames
