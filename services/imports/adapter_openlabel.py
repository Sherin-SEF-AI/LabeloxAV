"""ASAM OpenLABEL import: inverts services.export.adapter_openlabel. Global objects carry the
ontology type + typed attributes; per-frame object_data carries the bbox (cx,cy,w,h absolute). Image
references come from each frame's stream uri; frame width/height are read from the image at write time.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports._util import find_file, load_json
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_openlabel")


def _attrs(object_data: dict) -> dict:
    out: dict = {}
    for kind in ("text", "num", "boolean"):
        for a in object_data.get(kind, []):
            out[a["name"]] = a["val"]
    return out


def parse(root: Path) -> list[ImportFrame]:
    path = find_file(root, "openlabel.json", "*.json")
    if path is None:
        raise FileNotFoundError("no openlabel.json found under the dataset")
    doc = load_json(path)
    ol = doc.get("openlabel") if isinstance(doc, dict) else None
    if not ol or "frames" not in ol:
        raise ValueError("not an OpenLABEL document (missing openlabel.frames)")

    globals_ = ol.get("objects", {})
    types = {uid: o.get("type", "object_fallback") for uid, o in globals_.items()}

    frames: list[ImportFrame] = []
    for _, fr in sorted(ol["frames"].items(), key=lambda kv: int(kv[0])):
        props = fr.get("frame_properties", {})
        streams = props.get("streams", {})
        cam = next(iter(streams), "cam_front")
        ref = streams.get(cam, {}).get("uri")
        if not ref:
            continue
        objs: list[ImportObject] = []
        for uid, ob in fr.get("objects", {}).items():
            bboxes = ob.get("object_data", {}).get("bbox", [])
            if not bboxes:
                continue
            cx, cy, w, h = bboxes[0]["val"]
            objs.append(ImportObject(
                name=types.get(uid, "object_fallback"),
                bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                attrs=_attrs(globals_.get(uid, {}).get("object_data", {})),
                track_ref=uid,
            ))
        frames.append(ImportFrame(image_ref=ref, cam_id=cam, ts_ns=props.get("timestamp"), objects=objs))
    log.info("import_openlabel.parsed", frames=len(frames))
    return frames
