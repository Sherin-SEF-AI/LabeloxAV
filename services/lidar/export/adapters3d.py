"""3D dataset export, extending the seal-and-export driver to the 3D data product. A 3D slice of object_3d is
sealed into a content-addressed dataset commit and written to OpenLABEL (primary), nuScenes (AV buyers),
KITTI, and Waymo, with a lossless JSON provenance sidecar; raw clouds export to LAS and PCD. The HD map
exports to Lanelet2 and OpenDRIVE through the existing pipeline. Every export pins a dataset commit and
records the box source, model, and calibration (provenance is one walk).
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sqlalchemy import select

from core.logging import get_logger
from db.models import DatasetCommit, Object3D, PointCloud
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.lidar.ingest.store import load_cloud

log = get_logger("lidar_export3d")


class Slice3D(BaseModel):
    session_ids: list[uuid.UUID] = []
    states: list[str] = ["accepted", "auto_accept"]
    class_names: list[str] = []
    min_conf: float = 0.0


async def fetch_3d_records(spec: Slice3D) -> list[dict]:
    """Materialize the 3D objects in the slice, joined to their cloud and session, with provenance."""
    onto = get_ontology()
    async with get_sessionmaker()() as db:
        q = (select(Object3D, PointCloud.ts_ns, PointCloud.cloud_uri, PointCloud.source, DbSession.vehicle_id,
                    DbSession.city)
             .join(PointCloud, Object3D.cloud_id == PointCloud.cloud_id)
             .join(DbSession, PointCloud.session_id == DbSession.session_id)
             .where(Object3D.conf >= spec.min_conf))
        if spec.states:
            q = q.where(Object3D.state.in_(spec.states))
        if spec.session_ids:
            q = q.where(PointCloud.session_id.in_(spec.session_ids))
        rows = (await db.execute(q.order_by(PointCloud.ts_ns))).all()
    recs = []
    for o, ts_ns, cloud_uri, src, vehicle, city in rows:
        name = onto.by_id(o.class_id).name
        if spec.class_names and name not in spec.class_names:
            continue
        recs.append({"object_3d_id": str(o.object_3d_id), "cloud_id": str(o.cloud_id), "ts_ns": int(ts_ns),
                     "cloud_uri": cloud_uri, "cloud_source": src, "vehicle_id": vehicle, "city": city,
                     "class_id": o.class_id, "class_name": name, "center": o.center, "dims": o.dims,
                     "yaw": o.yaw, "pitch": o.pitch, "roll": o.roll, "conf": o.conf, "state": o.state,
                     "box_source": o.box_source, "track_3d_id": str(o.track_3d_id) if o.track_3d_id else None,
                     "object_id": str(o.object_id) if o.object_id else None,
                     "provenance": {"box_source": o.box_source, "source": o.source, **(o.provenance or {})}})
    return recs


def seal_3d_commit_id(spec: Slice3D, records: list[dict], ontology_version: str) -> str:
    """Deterministic content-addressed commit id over the 3D object membership."""
    h = hashlib.sha256()
    h.update(json.dumps(spec.model_dump(), sort_keys=True, default=str).encode())
    h.update(ontology_version.encode())
    for oid in sorted(r["object_3d_id"] for r in records):
        h.update(oid.encode())
    return f"lbx3d-{h.hexdigest()[:16]}"


def _yaw_quat(yaw: float) -> list[float]:
    """A yaw about the up axis to a [w, x, y, z] quaternion (nuScenes order)."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def write_openlabel_3d(records: list[dict], out_dir: Path) -> Path:
    """OpenLABEL JSON with a cuboid per object (9-DOF: x y z, rx ry rz, sx sy sz). Carries the class,
    confidence, box source, and relations (the linked 2D object and 3D track)."""
    objects = {}
    for r in records:
        objects[r["object_3d_id"]] = {
            "name": r["object_3d_id"], "type": r["class_name"],
            "object_data": {"cuboid": [{"name": "shape", "val": [
                r["center"][0], r["center"][1], r["center"][2], r["roll"], r["pitch"], r["yaw"],
                r["dims"][1], r["dims"][0], r["dims"][2]]}],   # sx,sy,sz = width,length,height
                "num": [{"name": "confidence", "val": r["conf"]}],
                "text": [{"name": "box_source", "val": r["box_source"]},
                         {"name": "cloud_id", "val": r["cloud_id"]},
                         {"name": "linked_2d_object", "val": r["object_id"] or ""},
                         {"name": "track_3d", "val": r["track_3d_id"] or ""}]}}
    doc = {"openlabel": {"metadata": {"schema_version": "1.0.0", "annotator": "labelox-3d"}, "objects": objects}}
    path = out_dir / "openlabel_3d.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def write_nuscenes_3d(records: list[dict], out_dir: Path) -> Path:
    """nuScenes-style sample_annotation: translation, size [w,l,h], rotation quaternion."""
    anns = []
    for r in records:
        anns.append({"sample_annotation_token": r["object_3d_id"], "instance_token": r["object_id"] or r["object_3d_id"],
                     "category_name": r["class_name"], "translation": list(r["center"]),
                     "size": [r["dims"][1], r["dims"][0], r["dims"][2]], "rotation": _yaw_quat(r["yaw"]),
                     "num_lidar_pts": r["provenance"].get("n_points", 0), "confidence": r["conf"]})
    path = out_dir / "nuscenes_3d.json"
    path.write_text(json.dumps({"sample_annotation": anns}, indent=2))
    return path


def write_kitti_3d(records: list[dict], out_dir: Path) -> Path:
    """KITTI label3d lines: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y. The 2D bbox is
    unknown (this is a LiDAR export); the 3D fields carry the ego cuboid."""
    by_cloud: dict[str, list[str]] = {}
    for r in records:
        h_, w_, l_ = r["dims"][2], r["dims"][1], r["dims"][0]
        x, y, z = r["center"]
        line = (f"{r['class_name']} -1 -1 -10 -1 -1 -1 -1 "
                f"{h_:.2f} {w_:.2f} {l_:.2f} {x:.2f} {y:.2f} {z:.2f} {r['yaw']:.4f} {r['conf']:.2f}")
        by_cloud.setdefault(r["cloud_id"], []).append(line)
    d = out_dir / "kitti_3d"
    d.mkdir(parents=True, exist_ok=True)
    for cloud_id, lines in by_cloud.items():
        (d / f"{cloud_id}.txt").write_text("\n".join(lines))
    return d


def write_waymo_3d(records: list[dict], out_dir: Path) -> Path:
    """Waymo-style laser labels (JSON): box {center_x/y/z, length, width, height, heading}, type, id."""
    frames: dict[str, list[dict]] = {}
    for r in records:
        frames.setdefault(r["cloud_id"], []).append({
            "id": r["object_3d_id"], "type": r["class_name"],
            "box": {"center_x": r["center"][0], "center_y": r["center"][1], "center_z": r["center"][2],
                    "length": r["dims"][0], "width": r["dims"][1], "height": r["dims"][2], "heading": r["yaw"]},
            "score": r["conf"]})
    path = out_dir / "waymo_3d.json"
    path.write_text(json.dumps({"frames": [{"context_name": k, "laser_labels": v} for k, v in frames.items()]},
                               indent=2))
    return path


def write_las(cloud, path: Path) -> Path:
    """Raw cloud to LAS via laspy (intensity scaled to 16-bit)."""
    import laspy

    las = laspy.LasData(laspy.LasHeader(point_format=3))
    las.x, las.y, las.z = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    las.intensity = np.clip(cloud.intensity * 65535.0, 0, 65535).astype(np.uint16)
    las.write(str(path))
    return path


def write_pcd(cloud, path: Path) -> Path:
    """Raw cloud to PCD via pypcd4 (xyz + intensity)."""
    from pypcd4 import PointCloud as PCD

    pts = np.concatenate([cloud.xyz, cloud.intensity.reshape(-1, 1)], axis=1).astype(np.float32)
    PCD.from_xyzi_points(pts).save(str(path))
    return path


_FORMAT_WRITERS = {"openlabel": write_openlabel_3d, "nuscenes": write_nuscenes_3d,
                   "kitti": write_kitti_3d, "waymo": write_waymo_3d}


async def export_3d_dataset(spec: Slice3D, formats: list[str] | None = None, out_root: Path | None = None,
                            export_clouds: bool = False) -> dict:
    """Seal a 3D slice into a dataset commit and write it to each requested format with a provenance sidecar."""
    formats = formats or ["openlabel", "nuscenes", "kitti", "waymo"]
    import tempfile

    onto = get_ontology()
    records = await fetch_3d_records(spec)
    commit_id = seal_3d_commit_id(spec, records, onto.version)
    out_root = Path(out_root) if out_root else Path(tempfile.gettempdir()) / "lbx_export3d"
    out_dir = out_root / commit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    export_uris: dict[str, str] = {}
    for fmt in formats:
        writer = _FORMAT_WRITERS.get(fmt)
        if writer is not None:
            export_uris[fmt] = str(writer(records, out_dir))

    # lossless provenance sidecar
    (out_dir / "provenance.json").write_text(json.dumps(
        {"commit_id": commit_id, "ontology_version": onto.version, "spec": spec.model_dump(mode="json"),
         "records": records}, indent=2, default=str))
    export_uris["provenance"] = str(out_dir / "provenance.json")

    cloud_ids = sorted({r["cloud_id"] for r in records})
    if export_clouds:
        cloud_dir = out_dir / "clouds"
        cloud_dir.mkdir(exist_ok=True)
        seen = {}
        for r in records:
            if r["cloud_id"] in seen:
                continue
            seen[r["cloud_id"]] = True
            cloud = load_cloud(r["cloud_uri"])
            write_las(cloud, cloud_dir / f"{r['cloud_id']}.las")
            write_pcd(cloud, cloud_dir / f"{r['cloud_id']}.pcd")
        export_uris["clouds_las_pcd"] = str(cloud_dir)

    async with get_sessionmaker()() as db:
        existing = await db.get(DatasetCommit, commit_id)
        if existing is None:
            db.add(DatasetCommit(commit_id=commit_id, slice_spec=spec.model_dump(mode="json"),
                                 object_count=0, object_3d_count=len(records), cloud_count=len(cloud_ids),
                                 ontology_version=onto.version, export_uris=export_uris,
                                 notes="lidar 3D export"))
        else:
            existing.object_3d_count = len(records)
            existing.cloud_count = len(cloud_ids)
            existing.export_uris = export_uris
        await db.commit()
    log.info("lidar.export3d", commit=commit_id, objects=len(records), clouds=len(cloud_ids),
             formats=list(export_uris.keys()))
    return {"commit_id": commit_id, "object_3d_count": len(records), "cloud_count": len(cloud_ids),
            "formats": list(export_uris.keys()), "export_uris": export_uris, "out_dir": str(out_dir)}
