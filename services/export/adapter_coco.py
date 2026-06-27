"""COCO adapter: boxes + polygon masks + categories. The fields COCO cannot natively carry
(typed attributes, track ids, full provenance) are preserved in an extension block on each
annotation, and the Parquet sidecar remains the lossless source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from core.storage import ObjectStore
from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

log = get_logger("adapter_coco")


def _load_polygons(store: ObjectStore, mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    try:
        data = json.loads(store.get_bytes(mask_uri))
        return data.get("polygons", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("coco.mask_load_failed", uri=mask_uri, error=str(exc))
        return []


def write_coco(
    records: list[ExportRecord],
    onto: Ontology,
    store: ObjectStore,
    out_dir: Path,
    filename: str = "annotations.json",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": c.id, "name": c.name, "supercategory": c.l1, "india_specific": c.india}
        for c in sorted(onto.classes, key=lambda c: c.id)
    ]

    images: dict[str, dict] = {}
    image_ids: dict[str, int] = {}
    annotations: list[dict] = []

    for i, r in enumerate(records, start=1):
        fkey = str(r.frame_id)
        if fkey not in image_ids:
            img_id = len(image_ids) + 1
            image_ids[fkey] = img_id
            images[fkey] = {
                "id": img_id,
                "file_name": r.img_uri.split("/")[-1],
                "uri": r.img_uri,
                "width": r.width,
                "height": r.height,
                "ts_ns": r.ts_ns,
                "session_id": str(r.session_id),
                "cam_id": r.cam_id,
            }
        x1, y1, x2, y2 = r.bbox
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        polys = _load_polygons(store, r.mask_uri)
        annotations.append(
            {
                "id": i,
                "image_id": image_ids[fkey],
                "category_id": r.class_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": polys,
                # extension block: what COCO cannot natively express, kept for fidelity
                "labelox": {
                    "object_id": str(r.object_id),
                    "track_id": str(r.track_id) if r.track_id else None,
                    "class_name": r.class_name,
                    "conf": r.conf,
                    "state": r.state,
                    "source": r.source,
                    "attributes": r.attrs,
                    "provenance": r.provenance,
                },
            }
        )

    coco = {
        "info": {"description": "LabeloxAV export", "ontology_version": onto.version},
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories,
    }
    path = out_dir / filename
    path.write_text(json.dumps(coco, indent=2))
    return path
