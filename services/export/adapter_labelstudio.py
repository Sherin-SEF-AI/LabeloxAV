"""Label Studio JSON export.

Mirrors services/imports/adapter_labelstudio.py, so the pair round-trips.

Geometry is emitted in PERCENT of image size, which is what Label Studio expects and what the importer here
converts back from. Writing pixels instead would load without error and place every box in the top-left corner,
so original_width/original_height are always emitted alongside: they are what makes the percentages meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from core.storage import ObjectStore
from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

log = get_logger("export_labelstudio")


def _load_polygons(store: ObjectStore, mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    try:
        blob = json.loads(store.get_bytes(mask_uri))
        return blob.get("polygons") or []
    except Exception as exc:  # noqa: BLE001
        log.warning("export_labelstudio.mask_read_failed", uri=mask_uri, error=str(exc))
        return []


def write_labelstudio(records: list[ExportRecord], onto: Ontology, store: ObjectStore, out_dir: Path,
                      filename: str = "tasks.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_frame.setdefault(str(r.frame_id), []).append(r)

    tasks = []
    for fid, recs in sorted(by_frame.items()):
        first = recs[0]
        W, H = float(first.width or 1), float(first.height or 1)
        results = []
        for r in recs:
            rid = str(r.object_id)
            polys = _load_polygons(store, r.mask_uri)
            if polys:
                for pi, poly in enumerate(polys):
                    pts = [[poly[i] / W * 100.0, poly[i + 1] / H * 100.0]
                           for i in range(0, len(poly) - 1, 2)]
                    if len(pts) < 3:
                        continue
                    results.append({
                        "id": f"{rid}-p{pi}", "type": "polygonlabels",
                        "from_name": "label", "to_name": "image",
                        "original_width": int(W), "original_height": int(H),
                        "image_rotation": 0,
                        "value": {"points": pts, "polygonlabels": [r.class_name]},
                    })
            else:
                x1, y1, x2, y2 = r.bbox
                results.append({
                    "id": rid, "type": "rectanglelabels",
                    "from_name": "label", "to_name": "image",
                    "original_width": int(W), "original_height": int(H),
                    "image_rotation": 0,
                    "value": {
                        "x": x1 / W * 100.0, "y": y1 / H * 100.0,
                        "width": (x2 - x1) / W * 100.0, "height": (y2 - y1) / H * 100.0,
                        "rotation": r.rot_deg or 0.0,
                        "rectanglelabels": [r.class_name],
                    },
                })

        tasks.append({
            "id": fid,
            "data": {"image": first.img_uri},
            # Exported as a human annotation set, matching how the importer reads it back.
            "annotations": [{"result": results, "ground_truth": False}],
            "meta": {"frame_id": fid, "session_id": str(first.session_id),
                     "cam_id": first.cam_id, "ts_ns": first.ts_ns},
        })

    path = out_dir / filename
    path.write_text(json.dumps(tasks, indent=2))
    log.info("export_labelstudio.written", tasks=len(tasks), objects=len(records), path=str(path))
    return path
