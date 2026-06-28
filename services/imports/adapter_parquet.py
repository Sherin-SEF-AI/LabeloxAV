"""Parquet import: the lossless inverse of services.export.adapter_parquet. Because the Parquet
sidecar carries every Unified Object field (ontology class id + name, full bbox, attrs, provenance,
track), re-importing it reconstructs the objects exactly. This is the free correctness oracle for the
round-trip test (export -> import -> counts + class names match). Images come from the stored img_uri.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from core.logging import get_logger
from services.imports._util import find_file
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_parquet")


def parse(root: Path) -> list[ImportFrame]:
    path = find_file(root, "objects.parquet", "*.parquet")
    if path is None:
        raise FileNotFoundError("no objects.parquet found under the dataset")
    table = pq.read_table(path).to_pylist()

    by_frame: dict[str, ImportFrame] = {}
    for r in table:
        fkey = r["frame_id"]
        if fkey not in by_frame:
            by_frame[fkey] = ImportFrame(
                image_ref=r["img_uri"], width=r.get("width"), height=r.get("height"),
                ts_ns=r.get("ts_ns"), cam_id=r.get("cam_id", "cam_front"), objects=[],
            )
        by_frame[fkey].objects.append(ImportObject(
            name=r["class_name"],
            bbox=[r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]],
            attrs=json.loads(r["attrs_json"]) if r.get("attrs_json") else {},
            track_ref=r.get("track_id"),
            conf=float(r.get("conf") or 1.0),
            ontology_class_id=r.get("class_id"),
            provenance=json.loads(r["provenance_json"]) if r.get("provenance_json") else {},
            mask_uri=r.get("mask_uri"),
            mask_encoding=r.get("mask_encoding"),
            rot_deg=float(r.get("rot_deg") or 0.0),
            keypoints=json.loads(r["keypoints_json"]) if r.get("keypoints_json") else None,
        ))
    frames = list(by_frame.values())
    log.info("import_parquet.parsed", frames=len(frames), objects=len(table))
    return frames
