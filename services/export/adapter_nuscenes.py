"""nuScenes adapter. Emits the canonical nuScenes relational tables (category, attribute, sensor,
calibrated_sensor, ego_pose, log, scene, sample, sample_data, instance, sample_annotation,
visibility) with token-based foreign keys.

Honest scope: nuScenes is a 3D, multi-sensor, ego-pose-calibrated format. This build is 2D
single-camera, so 3D fields (translation/size/rotation, camera_intrinsic, ego pose) are emitted as
identity placeholders and the real 2D box rides in a non-standard `lbx_bbox2d` field. A LIMITATIONS
note ships alongside, and the Parquet provenance sidecar remains the lossless source of truth. Full
fidelity arrives with the LiDAR + calibration seam.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]
ZERO3 = [0.0, 0.0, 0.0]


def _box3d(cuboid: dict | None) -> tuple[list[float], list[float], list[float], bool]:
    """nuScenes (translation, size, rotation, is_real) from an ego-frame cuboid {center,size,yaw}.
    rotation is the yaw quaternion [w,x,y,z] about the vertical axis. Returns placeholders when absent."""
    if not cuboid or not cuboid.get("center") or not cuboid.get("size"):
        return ZERO3, ZERO3, IDENTITY_QUAT, False
    center = [round(float(x), 3) for x in cuboid["center"]]
    size = [round(float(x), 3) for x in cuboid["size"]]
    yaw = float(cuboid.get("yaw", 0.0))
    rotation = [round(math.cos(yaw / 2), 6), 0.0, 0.0, round(math.sin(yaw / 2), 6)]
    return center, size, rotation, True


def _tok(kind: str, key: str) -> str:
    return hashlib.md5(f"{kind}:{key}".encode()).hexdigest()


def _attr_tokens_table(onto: Ontology) -> tuple[list[dict], dict[str, str]]:
    table, by_name = [], {}
    for name, a in onto.attributes.items():
        tok = _tok("attribute", name)
        by_name[name] = tok
        table.append({"token": tok, "name": name, "description": f"LabeloxAV ontology attribute ({a.type})"})
    return table, by_name


def build_nuscenes(records: list[ExportRecord], onto: Ontology) -> dict[str, list]:
    category = [
        {"token": _tok("category", c.name), "name": c.name, "description": f"{c.l0}/{c.l1}", "index": c.id}
        for c in sorted(onto.classes, key=lambda c: c.id)
    ]
    cat_tok = {c.name: _tok("category", c.name) for c in onto.classes}
    attribute, attr_tok = _attr_tokens_table(onto)
    visibility = [
        {"token": str(i), "level": lvl, "description": desc}
        for i, (lvl, desc) in enumerate(
            [("v0-40", "0-40%"), ("v40-60", "40-60%"), ("v60-80", "60-80%"), ("v80-100", "80-100%")], start=1
        )
    ]

    sensor, calibrated_sensor, ego_pose = [], [], []
    log_t, scene_t, sample_t, sample_data_t = [], [], [], []
    instance_t, annotation_t = [], []
    seen_sensor: set[str] = set()
    seen_cs: set[str] = set()

    # Group records by session (scene) and by frame (sample), preserving ts order.
    by_session: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_session.setdefault(str(r.session_id), []).append(r)

    # Instances: group annotations by track when available, else one instance per object.
    inst_members: dict[str, list[ExportRecord]] = {}
    for r in records:
        ikey = str(r.track_id) if r.track_id else f"obj:{r.object_id}"
        inst_members.setdefault(ikey, []).append(r)

    for sess, recs in by_session.items():
        recs_sorted = sorted(recs, key=lambda r: r.ts_ns)
        log_tok = _tok("log", sess)
        scene_tok = _tok("scene", sess)
        veh = recs_sorted[0].vehicle_id
        city = recs_sorted[0].city or "unknown"
        log_t.append({"token": log_tok, "logfile": f"session-{sess[:8]}", "vehicle": veh,
                      "date_captured": "", "location": city})

        # one sample per unique frame (key frames)
        frame_order: list[str] = []
        frame_first: dict[str, ExportRecord] = {}
        for r in recs_sorted:
            fk = str(r.frame_id)
            if fk not in frame_first:
                frame_first[fk] = r
                frame_order.append(fk)

        sample_tokens = [str(r.frame_id).replace("-", "") for r in (frame_first[fk] for fk in frame_order)]
        for i, fk in enumerate(frame_order):
            r = frame_first[fk]
            cam = r.cam_id
            sensor_tok = _tok("sensor", cam)
            if sensor_tok not in seen_sensor:
                seen_sensor.add(sensor_tok)
                sensor.append({"token": sensor_tok, "channel": cam.upper(), "modality": "camera"})
            cs_tok = _tok("cs", f"{sess}:{cam}")
            if cs_tok not in seen_cs:
                seen_cs.add(cs_tok)
                calibrated_sensor.append({"token": cs_tok, "sensor_token": sensor_tok,
                                          "translation": ZERO3, "rotation": IDENTITY_QUAT, "camera_intrinsic": []})
            ego_tok = _tok("ego", fk)
            ego_pose.append({"token": ego_tok, "timestamp": r.ts_ns // 1000,
                             "translation": ZERO3, "rotation": IDENTITY_QUAT})

            stok = sample_tokens[i]
            sample_t.append({
                "token": stok, "timestamp": r.ts_ns // 1000, "scene_token": scene_tok,
                "prev": sample_tokens[i - 1] if i > 0 else "",
                "next": sample_tokens[i + 1] if i < len(frame_order) - 1 else "",
            })
            sd_tok = _tok("sd", fk)
            sample_data_t.append({
                "token": sd_tok, "sample_token": stok, "ego_pose_token": ego_tok,
                "calibrated_sensor_token": cs_tok, "filename": r.img_uri, "fileformat": "jpg",
                "width": r.width, "height": r.height, "timestamp": r.ts_ns // 1000,
                "is_key_frame": True, "prev": "", "next": "",
            })

        scene_t.append({
            "token": scene_tok, "name": f"scene-{sess[:8]}", "description": f"{veh} {city}",
            "log_token": log_tok, "nbr_samples": len(frame_order),
            "first_sample_token": sample_tokens[0] if sample_tokens else "",
            "last_sample_token": sample_tokens[-1] if sample_tokens else "",
        })

    # Instances + annotations
    for ikey, members in inst_members.items():
        members_sorted = sorted(members, key=lambda r: r.ts_ns)
        inst_tok = _tok("inst", ikey)
        ann_tokens = [_tok("ann", str(r.object_id)) for r in members_sorted]
        instance_t.append({
            "token": inst_tok, "category_token": cat_tok[members_sorted[0].class_name],
            "nbr_annotations": len(members_sorted),
            "first_annotation_token": ann_tokens[0], "last_annotation_token": ann_tokens[-1],
        })
        for j, r in enumerate(members_sorted):
            attr_tokens = [attr_tok[k] for k, v in (r.attrs or {}).items() if v is True and k in attr_tok]
            occ = (r.attrs or {}).get("occlusion")
            vis = {0: "4", 25: "3", 50: "2", 75: "1", 100: "1"}.get(occ, "4")
            translation, size, rotation, has3d = _box3d(r.cuboid_3d)
            annotation_t.append({
                "token": ann_tokens[j], "sample_token": str(r.frame_id).replace("-", ""),
                "instance_token": inst_tok, "visibility_token": vis, "attribute_tokens": attr_tokens,
                "translation": translation, "size": size, "rotation": rotation, "lbx_has_3d": has3d,
                "num_lidar_pts": 0, "num_radar_pts": 0,
                "prev": ann_tokens[j - 1] if j > 0 else "", "next": ann_tokens[j + 1] if j < len(members_sorted) - 1 else "",
                "lbx_bbox2d": [round(v, 2) for v in r.bbox],  # non-standard: the real 2D box
                "lbx_class": r.class_name, "lbx_conf": r.conf,
            })

    return {
        "category": category, "attribute": attribute, "visibility": visibility, "sensor": sensor,
        "calibrated_sensor": calibrated_sensor, "ego_pose": ego_pose, "log": log_t, "scene": scene_t,
        "sample": sample_t, "sample_data": sample_data_t, "instance": instance_t,
        "sample_annotation": annotation_t,
    }


def write_nuscenes(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    tables = build_nuscenes(records, onto)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2))
    n3d = sum(1 for r in records if r.cuboid_3d)
    (out_dir / "LIMITATIONS.md").write_text(
        "# nuScenes export scope\n\n"
        "nuScenes table shape over a single camera. Each sample_annotation carries the real 2D box in the "
        "non-standard `lbx_bbox2d` field. When an object has a 3D cuboid label its translation/size/rotation "
        "are real (ego-frame metres, yaw quaternion) and `lbx_has_3d` is true; otherwise they are identity "
        f"placeholders. This export has {n3d} of {len(records)} annotations with a real 3D box. Global ego "
        "pose and calibrated sensors arrive with the LiDAR + calibration seam. The Parquet provenance "
        "sidecar is the lossless source of truth.\n"
    )
    return out_dir
