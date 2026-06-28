"""BDD100K import: inverts services.export.adapter_bdd. Reads `bdd_det.json` (a list of per-image
entries with a `labels` array) into ImportFrame[]. Each label's `category` -> ImportObject.name and
its `box2d` {x1,y1,x2,y2} -> xyxy pixel bbox. Our own exports carry a `labelox` extension block
(track id, conf, state, source, provenance) used verbatim for a lossless round-trip; foreign BDD
falls back to the category + box + attributes.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports._util import find_file, load_json
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_bdd")


def parse(root: Path) -> list[ImportFrame]:
    path = find_file(root, "bdd_det.json", "*.json")
    if path is None:
        raise FileNotFoundError("no BDD bdd_det.json found under the dataset")
    doc = load_json(path)
    if not isinstance(doc, list):
        raise ValueError(f"{path.name} is not a BDD detection file (expected a list of images)")

    frames: list[ImportFrame] = []
    for entry in doc:
        if "box2d" in entry or "labels" not in entry:
            raise ValueError(f"{path.name} is not a BDD detection file (missing per-image labels)")
        objs: list[ImportObject] = []
        for lab in entry.get("labels", []):
            box = lab.get("box2d")
            if not box:
                continue  # poly2d / lane / drivable-area labels carry no detection box
            bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
            ext = lab.get("labelox") or {}
            objs.append(ImportObject(
                name=lab.get("category", "object_fallback"), bbox=bbox,
                attrs=lab.get("attributes", {}), track_ref=ext.get("track_id"),
                conf=float(ext.get("conf", 1.0)), provenance=ext.get("provenance", {}),
            ))
        ref = entry.get("name")
        if not ref:
            continue
        frames.append(ImportFrame(image_ref=ref, objects=objs))
    log.info("import_bdd.parsed", frames=len(frames), file=path.name)
    return frames
