"""ASAM OpenLABEL adapter: the near-lossless strategic export for serious AV buyers. Objects are
defined globally with their ontology type; per-frame object_data carries the bbox (cx,cy,w,h) and
poly2d mask, and typed attributes ride natively in the OpenLABEL attribute block (text/num/boolean).
Provenance and ontology version live in metadata, so the export is near-lossless on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from core.storage import ObjectStore
from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

log = get_logger("adapter_openlabel")


def _load_polygons(store: ObjectStore, mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    try:
        return json.loads(store.get_bytes(mask_uri)).get("polygons", [])
    except Exception:
        return []


def _attribute_block(rec: ExportRecord, onto: Ontology) -> dict:
    text, num, boolean = [], [], []
    text.append({"name": "state", "val": rec.state})
    text.append({"name": "source", "val": rec.source})
    num.append({"name": "confidence", "val": rec.conf})
    if rec.track_id:
        text.append({"name": "track_id", "val": str(rec.track_id)})
    for k, v in (rec.attrs or {}).items():
        if isinstance(v, bool):
            boolean.append({"name": k, "val": v})
        elif isinstance(v, (int, float)):
            num.append({"name": k, "val": v})
        else:
            text.append({"name": k, "val": json.dumps(v) if isinstance(v, (list, dict)) else str(v)})
    block: dict = {}
    if text:
        block["text"] = text
    if num:
        block["num"] = num
    if boolean:
        block["boolean"] = boolean
    return block


def write_openlabel(
    records: list[ExportRecord],
    onto: Ontology,
    store: ObjectStore,
    out_dir: Path,
    filename: str = "openlabel.json",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_global: dict[str, dict] = {}
    frames: dict[str, dict] = {}
    frame_index: dict[str, int] = {}
    frame_intervals: list[int] = []

    for rec in records:
        uid = str(rec.object_id)
        fkey = str(rec.frame_id)
        if fkey not in frame_index:
            idx = len(frame_index)
            frame_index[fkey] = idx
            frames[str(idx)] = {
                "frame_properties": {
                    "timestamp": rec.ts_ns,
                    "streams": {rec.cam_id: {"uri": rec.img_uri}},
                },
                "objects": {},
            }
            frame_intervals.append(idx)

        fi = frame_index[fkey]
        x1, y1, x2, y2 = rec.bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        object_data: dict = {"bbox": [{"name": "shape", "val": [cx, cy, w, h]}]}
        polys = _load_polygons(store, rec.mask_uri)
        if polys:
            object_data["poly2d"] = [
                {"name": "mask", "val": poly, "mode": "MODE_POLY2D_ABSOLUTE", "closed": True}
                for poly in polys
            ]
        if rec.polyline and len(rec.polyline) >= 2:
            object_data.setdefault("poly2d", []).append(
                {"name": "polyline", "val": [c for pt in rec.polyline for c in pt],
                 "mode": "MODE_POLY2D_ABSOLUTE", "closed": False})
        frames[str(fi)]["objects"][uid] = {"object_data": object_data}

        if uid not in objects_global:
            objects_global[uid] = {
                "name": f"{rec.class_name}-{uid[:8]}",
                "type": rec.class_name,
                "object_data": _attribute_block(rec, onto),
                "frame_intervals": [{"frame_start": fi, "frame_end": fi}],
            }
        else:
            objects_global[uid]["frame_intervals"][0]["frame_end"] = fi

    doc = {
        "openlabel": {
            "metadata": {
                "schema_version": "1.0.0",
                "name": "LabeloxAV export",
                "annotator": "LabeloxAV auto-label + human review",
                "ontology_version": onto.version,
            },
            "ontologies": {"0": f"labelox:{onto.version}"},
            "frames": frames,
            "frame_intervals": (
                [{"frame_start": min(frame_intervals), "frame_end": max(frame_intervals)}]
                if frame_intervals
                else []
            ),
            "objects": objects_global,
        }
    }
    path = out_dir / filename
    path.write_text(json.dumps(doc, indent=2))
    return path
