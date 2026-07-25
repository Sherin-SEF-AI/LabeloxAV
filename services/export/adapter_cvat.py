"""CVAT XML export ("CVAT for images 1.1").

The other half of the migration path: work labeled here can go back into a CVAT instance, which matters when
a team runs both. Written to mirror services/imports/adapter_cvat.py exactly, so a round trip through the pair
is lossless for the fields CVAT can express.

Geometry that CVAT cannot represent is dropped rather than approximated, and the count of what was dropped is
returned. Silently degrading a 3D cuboid into a 2D box would produce a file that looks complete and is not.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from core.logging import get_logger
from core.storage import ObjectStore
from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

log = get_logger("export_cvat")


def _points_str(flat: list[float]) -> str:
    """[x,y,x,y,...] -> 'x,y;x,y' (the CVAT points encoding)."""
    return ";".join(f"{flat[i]:.2f},{flat[i + 1]:.2f}" for i in range(0, len(flat) - 1, 2))


def _load_polygons(store: ObjectStore, mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    try:
        import json

        blob = json.loads(store.get_bytes(mask_uri))
        return blob.get("polygons") or []
    except Exception as exc:  # noqa: BLE001
        log.warning("export_cvat.mask_read_failed", uri=mask_uri, error=str(exc))
        return []


def write_cvat(records: list[ExportRecord], onto: Ontology, store: ObjectStore, out_dir: Path,
               filename: str = "annotations.xml") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "labeloxav-export"
    labels_el = ET.SubElement(task, "labels")
    for c in sorted(onto.classes, key=lambda c: c.id):
        lab = ET.SubElement(labels_el, "label")
        ET.SubElement(lab, "name").text = c.name

    # group records by frame, since CVAT nests shapes under <image>
    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_frame.setdefault(str(r.frame_id), []).append(r)

    dropped = 0
    for idx, (fid, recs) in enumerate(sorted(by_frame.items())):
        first = recs[0]
        img = ET.SubElement(root, "image", {
            "id": str(idx), "name": f"{fid}.jpg",
            "width": str(first.width), "height": str(first.height),
        })
        for r in recs:
            name = r.class_name
            polys = _load_polygons(store, r.mask_uri)
            if polys:
                for poly in polys:
                    if len(poly) >= 6:
                        ET.SubElement(img, "polygon", {
                            "label": name, "occluded": "0", "points": _points_str(poly), "z_order": "0"})
            elif r.polyline:
                flat = [c for p in r.polyline for c in p]
                ET.SubElement(img, "polyline", {
                    "label": name, "occluded": "0", "points": _points_str(flat), "z_order": "0"})
            else:
                x1, y1, x2, y2 = r.bbox
                attrs = {"label": name, "occluded": "0",
                         "xtl": f"{x1:.2f}", "ytl": f"{y1:.2f}",
                         "xbr": f"{x2:.2f}", "ybr": f"{y2:.2f}", "z_order": "0"}
                if r.rot_deg:
                    attrs["rotation"] = f"{r.rot_deg:.2f}"
                box = ET.SubElement(img, "box", attrs)
                for k, v in (r.attrs or {}).items():
                    a = ET.SubElement(box, "attribute", {"name": str(k)})
                    a.text = str(v)
            if r.cuboid_3d:
                dropped += 1     # CVAT for images has no 3D cuboid; counted, not silently flattened

    path = out_dir / filename
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    log.info("export_cvat.written", frames=len(by_frame), objects=len(records),
             cuboids_dropped=dropped, path=str(path))
    return path
