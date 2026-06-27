"""nuScenes import: inverts services.export.adapter_nuscenes. The real 2D box rides in the
non-standard `lbx_bbox2d` field (the export is 2D single-camera in nuScenes table shape), so import
reads that plus `lbx_class`. sample_data gives the image filename + dimensions per sample.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports._util import find_file, load_json
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_nuscenes")


def parse(root: Path) -> list[ImportFrame]:
    sd_path = find_file(root, "sample_data.json")
    ann_path = find_file(root, "sample_annotation.json")
    if sd_path is None or ann_path is None:
        raise FileNotFoundError("no nuScenes sample_data.json / sample_annotation.json found")

    sample_data = load_json(sd_path)
    annotations = load_json(ann_path)

    by_sample: dict[str, dict] = {sd["sample_token"]: sd for sd in sample_data}
    anns_by_sample: dict[str, list[dict]] = {}
    for a in annotations:
        anns_by_sample.setdefault(a["sample_token"], []).append(a)

    frames: list[ImportFrame] = []
    for stok, sd in by_sample.items():
        objs: list[ImportObject] = []
        for a in anns_by_sample.get(stok, []):
            box = a.get("lbx_bbox2d")
            if not box:
                continue
            objs.append(ImportObject(
                name=a.get("lbx_class", "object_fallback"), bbox=list(box),
                track_ref=a.get("instance_token"), conf=float(a.get("lbx_conf", 1.0)),
            ))
        frames.append(ImportFrame(
            image_ref=sd["filename"], width=sd.get("width"), height=sd.get("height"),
            ts_ns=(sd.get("timestamp") or 0) * 1000, objects=objs,
        ))
    log.info("import_nuscenes.parsed", frames=len(frames))
    return frames
