"""Mapillary Vistas (v2.0) import: converts the polygon/segmentation annotations to bounding boxes.

Mapillary's polygon JSON (v2.0/polygons/<key>.json) carries {width, height, objects:[{label, polygon}]}
with hierarchical labels like "object--vehicle--car" / "human--person--individual". Only foreground
instance classes (object--/human--/animal--) become boxes; "stuff" classes (construction--/nature--/
marking--) are segmentation-only and skipped. Each polygon's pixel extent is the box; the Mapillary
label is mapped to the LabeloxAV ontology (unmapped names fall through remap.py's fallbacks).

Note: Mapillary v1.2+ RGB images are already face/plate blurred by Mapillary; Gate A still runs on
import (idempotent on already-blurred regions).
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports._util import load_json
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_mapillary")

_INSTANCE_PREFIXES = ("object--", "human--", "animal--")

# High-value drivable classes mapped explicitly; everything else flows through remap.py (which routes
# vehicle-ish tokens to vehicle_fallback, the rest to object_fallback).
MAPILLARY_TO_ONTOLOGY = {
    "object--vehicle--car": "sedan",
    "object--vehicle--truck": "truck",
    "object--vehicle--bus": "bus",
    "object--vehicle--motorcycle": "motorcycle",
    "object--vehicle--bicycle": "cycle",
    "object--vehicle--trailer": "multi_axle_trailer",
    "object--vehicle--caravan": "vehicle_fallback",
    "object--vehicle--other-vehicle": "vehicle_fallback",
    "object--vehicle--on-rails": "vehicle_fallback",
    "human--person--individual": "pedestrian",
    "human--person--person-group": "pedestrian",
    "human--rider--bicyclist": "rider",
    "human--rider--motorcyclist": "rider",
    "human--rider--other-rider": "rider",
    "animal--ground-animal": "cattle",
    "object--animal--ground-animal": "cattle",
    "object--traffic-light": "traffic_signal",
    "object--traffic-sign--front": "traffic_sign",
    "object--traffic-sign--back": "traffic_sign",
    "object--traffic-sign--direction-front": "traffic_sign",
    "object--traffic-sign--direction-back": "traffic_sign",
    "object--street-light": "street_light",
    "object--banner": "hoarding",
    "object--billboard": "hoarding",
    "object--mailbox": "postbox",
    "object--phone-booth": "telephone_booth",
    "object--cctv-camera": "cctv_pole",
}

_IMG_EXT = (".jpg", ".jpeg", ".png")


def _bbox(polygon: list[list[float]]) -> list[float] | None:
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _find_image(stem: str, root: Path) -> Path | None:
    for ext in _IMG_EXT:
        # prefer an images/ dir, then anywhere
        hits = [p for p in root.rglob(f"{stem}{ext}") if "images" in p.parts] or sorted(root.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def parse(root: Path) -> list[ImportFrame]:
    polys = [p for p in root.rglob("*.json") if "polygons" in p.parts]
    if not polys:
        # staging may have flattened the layout; fall back to any json that looks like a polygon doc
        polys = [p for p in root.rglob("*.json")
                 if (lambda d: isinstance(d, dict) and "objects" in d and "width" in d)(_safe(p))]
    if not polys:
        raise FileNotFoundError("no Mapillary polygon JSON found under the dataset")

    frames: list[ImportFrame] = []
    for pj in polys:
        doc = _safe(pj)
        if not doc or "objects" not in doc:
            continue
        img = _find_image(pj.stem, root)
        if img is None:
            continue
        objs: list[ImportObject] = []
        for o in doc["objects"]:
            label = o.get("label", "")
            if not label.startswith(_INSTANCE_PREFIXES):
                continue
            bb = _bbox(o.get("polygon", []))
            if bb is None:
                continue
            name = MAPILLARY_TO_ONTOLOGY.get(label, label.split("--")[-1])
            objs.append(ImportObject(name=name, bbox=bb, attrs={"mapillary_label": label}))
        frames.append(ImportFrame(image_ref=str(img), width=doc.get("width"), height=doc.get("height"), objects=objs))
    log.info("import_mapillary.parsed", frames=len(frames))
    return frames


def _safe(path: Path):
    try:
        return load_json(path)
    except Exception:  # noqa: BLE001
        return None
