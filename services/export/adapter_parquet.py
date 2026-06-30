"""Parquet adapter: the lossless columnar interchange and the provenance sidecar that rides with
every legacy-format export (Principle 10). Nothing the Unified Object carries is dropped here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from services.export.records import ExportRecord


def write_parquet(records: list[ExportRecord], out_dir: Path, filename: str = "objects.parquet") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {
        "object_id": [], "frame_id": [], "session_id": [], "track_id": [],
        "ts_ns": [], "cam_id": [], "img_uri": [], "width": [], "height": [],
        "vehicle_id": [], "city": [], "class_id": [], "class_name": [],
        "bbox_x1": [], "bbox_y1": [], "bbox_x2": [], "bbox_y2": [],
        "conf": [], "state": [], "source": [], "mask_uri": [], "mask_encoding": [],
        "rot_deg": [], "keypoints_json": [], "polyline_json": [], "relationships_json": [],
        "attrs_json": [], "provenance_json": [],
    }
    for r in records:
        cols["object_id"].append(str(r.object_id))
        cols["frame_id"].append(str(r.frame_id))
        cols["session_id"].append(str(r.session_id))
        cols["track_id"].append(str(r.track_id) if r.track_id else None)
        cols["ts_ns"].append(r.ts_ns)
        cols["cam_id"].append(r.cam_id)
        cols["img_uri"].append(r.img_uri)
        cols["width"].append(r.width)
        cols["height"].append(r.height)
        cols["vehicle_id"].append(r.vehicle_id)
        cols["city"].append(r.city)
        cols["class_id"].append(r.class_id)
        cols["class_name"].append(r.class_name)
        cols["bbox_x1"].append(r.bbox[0])
        cols["bbox_y1"].append(r.bbox[1])
        cols["bbox_x2"].append(r.bbox[2])
        cols["bbox_y2"].append(r.bbox[3])
        cols["conf"].append(r.conf)
        cols["state"].append(r.state)
        cols["source"].append(r.source)
        cols["mask_uri"].append(r.mask_uri)
        cols["mask_encoding"].append(r.mask_encoding)
        cols["rot_deg"].append(float(r.rot_deg or 0.0))
        cols["keypoints_json"].append(json.dumps(r.keypoints) if r.keypoints else None)
        cols["polyline_json"].append(json.dumps(r.polyline) if r.polyline else None)
        cols["relationships_json"].append(json.dumps(r.relationships) if r.relationships else None)
        cols["attrs_json"].append(json.dumps(r.attrs))
        cols["provenance_json"].append(json.dumps(r.provenance))

    table = pa.table(cols)
    path = out_dir / filename
    pq.write_table(table, path)
    return path
