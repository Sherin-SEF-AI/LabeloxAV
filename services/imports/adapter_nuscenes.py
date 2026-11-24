"""nuScenes import. Two sources share the table layout and are told apart by what the annotations carry.

**A standard nuScenes release** (`v1.0-mini`, `v1.0-trainval`: sample, sample_data, sample_annotation,
instance, category, attribute, visibility, calibrated_sensor, ego_pose, sensor ...). Annotations there are
3D boxes in the global frame and nothing else, so for every camera keyframe each box of its sample is taken
global -> ego (that camera's ego_pose) -> sensor (its calibrated_sensor), its eight corners are projected
with the camera intrinsic, and the 2D box is the bounds of the projected hull clipped to the image. A box
entirely behind the camera, or whose hull misses the image, is not an object of that frame. One annotation
therefore becomes one object in every camera that sees it, all sharing the nuScenes instance token as
`track_ref`. The devkit is not needed; the math is in services.imports.nuscenes_tables.

**LabeloxAV's own export** (services.export.adapter_nuscenes), a 2D single-camera slice in nuScenes table
shape whose real box rides in the non-standard `lbx_bbox2d` field beside `lbx_class`. That path is read as
it always was.

What a standard import carries per object:

    name          the nuScenes category name ("vehicle.car"); services.imports.remap maps it to the ontology
    bbox          xyxy pixels, clipped to the image
    track_ref     the instance token (one physical object across samples and cameras)
    attrs         nuscenes_visibility (the level, "v60-80") and nuscenes_attributes (names, "vehicle.moving")
    provenance    cuboid_3d: the box as this codebase stores cuboids (Object.cuboid_3d), ego frame,
                  {center: [x, y, z], size: [width, length, height], yaw: rad}
                  nuscenes_box: the same box with its full rotation, in the global, ego and sensor frames
                  nuscenes: the source tokens, the capture time and the lidar/radar point counts

ImportObject has no cuboid field, which is why the cuboid sits in provenance: services.imports.run copies
provenance onto the stored object but does not yet lift `cuboid_3d` into Object.cuboid_3d.

A frame's ts_ns is its SAMPLE's timestamp, not the camera's own capture time. The six cameras of one sample
fire up to ~50 ms apart, and this codebase groups a rig's frames (20 ms tolerance) and links clouds to
frames by ts_ns, so the sample time is what makes a nuScenes sample one rig instant here. The capture time
is kept in provenance, and `frame_time="capture"` switches to it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.logging import get_logger
from services.imports._util import find_file, load_json
from services.imports.nuscenes_tables import (
    NuScenesTables,
    apply_pose,
    heading_of,
    invert_pose,
    load_tables,
    quat_canonical,
    quat_conjugate,
    quat_multiply,
    quats_to_matrices,
)
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_nuscenes")

# The provenance key holding the cuboid in Object.cuboid_3d's own shape.
CUBOID_KEY = "cuboid_3d"
# Geometry closer to the image plane than this (metres along the optical axis) is cut away before
# projecting. A box that straddles the camera is clipped against this plane rather than having its rear
# corners projected through the lens centre, which would mirror them onto the wrong side of the image.
NEAR_M = 0.1
# A projected box thinner than this in either direction has no visible area worth an object.
MIN_VISIBLE_PX = 1.0

# Corner sign pattern (x, y, z), index = 4*ix + 2*iy + iz: the order services.lidar.boxes.cuboid_corners uses.
_SIGNS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)],
                  dtype=np.float64)
# The 12 edges: corner pairs whose sign patterns differ on exactly one axis.
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


# --- projecting a cuboid to a 2D box ----------------------------------------------------------------------

def box_corners(center, size_wlh, rotation: np.ndarray) -> np.ndarray:
    """The 8 corners of a nuScenes box: `size` is [width, length, height], the box x axis is its length."""
    width, length, height = (float(v) for v in size_wlh)
    local = _SIGNS * np.array([length, width, height], dtype=np.float64)
    return np.asarray(center, dtype=np.float64).reshape(3) + local @ np.asarray(rotation, dtype=np.float64).T


def _convex_hull(points) -> list[tuple[float, float]]:
    """Andrew's monotone chain, counter-clockwise, collinear points dropped."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _clip_halfplane(poly: list[tuple[float, float]], axis: int, bound: float, keep_below: bool):
    """One Sutherland-Hodgman pass: keep the part of a convex polygon with coord <= bound (or >= bound)."""
    out: list[tuple[float, float]] = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        a_in = a[axis] <= bound if keep_below else a[axis] >= bound
        b_in = b[axis] <= bound if keep_below else b[axis] >= bound
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            other = a[1 - axis] + t * (b[1 - axis] - a[1 - axis])
            out.append((bound, other) if axis == 0 else (other, bound))
    return out


def project_corners_to_bbox(corners_cam, intrinsic, width: float, height: float,
                            near: float = NEAR_M) -> list[float] | None:
    """The xyxy pixel box of a cuboid given its 8 corners in the camera optical frame, or None if the
    camera does not see it.

    The cuboid is first cut by the plane z = near, so a box partly behind the camera contributes only its
    front part. The remaining vertices are projected with the 3x3 intrinsic; their convex hull is the
    cuboid's silhouette, which is intersected with the image rectangle [0, width] x [0, height], and the
    box is the bounds of that intersection. Clipping the silhouette, rather than clamping its bounding box,
    is what keeps a box that only crosses an image corner from being reported wider or taller than the part
    of it that is actually in view.
    """
    pts = np.asarray(corners_cam, dtype=np.float64).reshape(8, 3)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    front = pts[:, 2] > near
    if not front.any():
        return None
    keep = [pts[i] for i in range(8) if front[i]]
    if not front.all():
        for i, j in _EDGES:
            if front[i] != front[j]:
                a, b = pts[i], pts[j]
                t = (near - a[2]) / (b[2] - a[2])
                keep.append(a + t * (b - a))
    uvw = np.asarray(keep) @ k.T
    uv = uvw[:, :2] / uvw[:, 2:3]

    x1, y1 = float(uv[:, 0].min()), float(uv[:, 1].min())
    x2, y2 = float(uv[:, 0].max()), float(uv[:, 1].max())
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None                                   # the silhouette's own bounds miss the image
    if not (x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height):
        poly = _convex_hull(uv)
        for axis, bound, keep_below in ((0, 0.0, False), (0, float(width), True),
                                        (1, 0.0, False), (1, float(height), True)):
            if len(poly) < 3:
                return None
            poly = _clip_halfplane(poly, axis, bound, keep_below)
        if len(poly) < 3:
            return None
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if (x2 - x1) < MIN_VISIBLE_PX or (y2 - y1) < MIN_VISIBLE_PX:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


# --- LabeloxAV's own export -------------------------------------------------------------------------------

def _is_labelox_export(annotations: list[dict], table_dir: Path) -> bool:
    """Our export marks every annotation with `lbx_bbox2d`. An export of a slice with no objects has no
    annotation to mark, so the LIMITATIONS note the exporter writes beside the tables decides that case."""
    if any("lbx_bbox2d" in a for a in annotations):
        return True
    note = table_dir / "LIMITATIONS.md"
    try:
        return note.is_file() and note.read_text().startswith("# nuScenes export scope")
    except OSError:
        return False


def _labelox_cuboid(a: dict) -> dict | None:
    """The ego-frame cuboid a LabeloxAV export wrote into translation/size/rotation (identity ego pose, a
    pure yaw quaternion), back in Object.cuboid_3d shape. None for the identity placeholders."""
    if not a.get("lbx_has_3d"):
        return None
    try:
        center = [float(v) for v in a["translation"]]
        size = [float(v) for v in a["size"]]
        qw, _qx, _qy, qz = (float(v) for v in a["rotation"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(center) != 3 or len(size) != 3:
        return None
    return {"center": center, "size": size, "yaw": round(2.0 * math.atan2(qz, qw), 6)}


def _parse_labelox_export(sample_data: list[dict], annotations: list[dict]) -> list[ImportFrame]:
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
            cuboid = _labelox_cuboid(a)
            objs.append(ImportObject(
                name=a.get("lbx_class", "object_fallback"), bbox=list(box),
                track_ref=a.get("instance_token"), conf=float(a.get("lbx_conf", 1.0)),
                provenance={CUBOID_KEY: cuboid} if cuboid else {},
            ))
        frames.append(ImportFrame(
            image_ref=sd["filename"], width=sd.get("width"), height=sd.get("height"),
            ts_ns=(sd.get("timestamp") or 0) * 1000, objects=objs,
        ))
    return frames


# --- a standard nuScenes release --------------------------------------------------------------------------

def _image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) read from the image, for the rare sample_data row that does not state them."""
    try:
        import cv2

        img = cv2.imread(str(path))
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    return int(img.shape[1]), int(img.shape[0])


def _image_ref(root: Path, tables: NuScenesTables, filename: str) -> str:
    """The image reference relative to the import root, so services.imports.run resolves it by exact path.
    `filename` is relative to the dataroot (the tables' parent), which is the import root only when the
    dataset was not wrapped in another folder. Falls back to the raw filename, which the driver then looks
    up by basename under the root."""
    candidate = tables.dataroot / filename
    try:
        if candidate.is_file():
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        pass
    return filename


def _vec(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def _parse_standard(root: Path, tables: NuScenesTables, frame_time: str) -> list[ImportFrame]:
    if frame_time not in ("sample", "capture"):
        raise ValueError(f"frame_time must be 'sample' or 'capture', got {frame_time!r}")

    anns_by_sample: dict[str, list[dict]] = {}
    for a in tables.sample_annotation:
        anns_by_sample.setdefault(a["sample_token"], []).append(a)

    # Camera keyframes, grouped by sample so a sample's boxes are built once for all of its cameras.
    cams_by_sample: dict[str, list[tuple[dict, str]]] = {}
    dangling = 0
    for sd in tables.sample_data:
        if not sd.get("is_key_frame"):
            continue                                   # sweeps between keyframes carry no annotations
        cs = tables.calibrated_sensor.get(sd.get("calibrated_sensor_token", ""))
        sensor = tables.sensor.get(cs.get("sensor_token", "")) if cs else None
        if (cs is None or sensor is None or sd.get("ego_pose_token") not in tables.ego_pose
                or sd.get("sample_token") not in tables.sample):
            dangling += 1
            continue
        if sensor.get("modality") != "camera":
            continue
        cams_by_sample.setdefault(sd["sample_token"], []).append((sd, sensor["channel"]))

    frames: list[ImportFrame] = []
    n_objects = n_bad_boxes = n_uncalibrated = 0
    for stok, cam_rows in cams_by_sample.items():
        sample = tables.sample[stok]
        scene = tables.scene.get(sample.get("scene_token", ""), {})

        # The sample's boxes in the global frame: centres (N, 3), rotations (N, 3, 3), corners (N, 8, 3).
        anns = anns_by_sample.get(stok, [])
        if anns:
            try:
                centers_g = np.array([a["translation"] for a in anns], dtype=np.float64).reshape(-1, 3)
                sizes = np.array([a["size"] for a in anns], dtype=np.float64).reshape(-1, 3)
                quats = np.array([a["rotation"] for a in anns], dtype=np.float64).reshape(-1, 4)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"sample {stok}: a sample_annotation is missing translation/size/rotation "
                                 f"({exc})") from exc
            norms = np.linalg.norm(quats, axis=1)
            usable = (np.isfinite(norms) & (norms > 1e-12) & np.isfinite(centers_g).all(axis=1)
                      & np.isfinite(sizes).all(axis=1) & (sizes > 0).all(axis=1))
            n_bad_boxes += int((~usable).sum())
            idx = np.flatnonzero(usable)
            anns = [anns[i] for i in idx]
            centers_g, sizes, quats = centers_g[idx], sizes[idx], quats[idx]
        if anns:
            rot_g = quats_to_matrices(quats)
            local = _SIGNS[None, :, :] * sizes[:, None, [1, 0, 2]]            # [l, w, h] along box x, y, z
            corners_g = centers_g[:, None, :] + np.einsum("nij,nkj->nki", rot_g, local)

        for sd, channel in cam_rows:
            width, height = sd.get("width"), sd.get("height")
            if not width or not height:
                size = _image_size(tables.file_path(sd))
                width, height = size if size else (None, None)
            ts_us = sample["timestamp"] if frame_time == "sample" else sd["timestamp"]
            frame = ImportFrame(image_ref=_image_ref(root, tables, sd["filename"]),
                                width=int(width) if width else None, height=int(height) if height else None,
                                ts_ns=int(ts_us) * 1000, cam_id=channel)
            frames.append(frame)
            if not anns:
                continue

            cs = tables.calibrated_sensor[sd["calibrated_sensor_token"]]
            ego = tables.ego_pose[sd["ego_pose_token"]]
            intrinsic = np.asarray(cs.get("camera_intrinsic") or [], dtype=np.float64)
            if intrinsic.shape != (3, 3) or not width or not height:
                n_uncalibrated += 1                    # no way to place a 3D box on this image
                continue
            sensor_to_ego = tables.sensor_to_ego(sd)
            global_to_ego = invert_pose(tables.ego_to_global(sd))
            global_to_sensor = invert_pose(sensor_to_ego) @ global_to_ego
            corners_s = (corners_g.reshape(-1, 3) @ global_to_sensor[:3, :3].T
                         + global_to_sensor[:3, 3]).reshape(-1, 8, 3)
            ahead = (corners_s[:, :, 2] > NEAR_M).any(axis=1)

            q_ego_inv = quat_conjugate(ego["rotation"])
            q_cs_inv = quat_conjugate(cs["rotation"])
            for n in np.flatnonzero(ahead):
                bbox = project_corners_to_bbox(corners_s[n], intrinsic, width, height)
                if bbox is None:
                    continue
                a = anns[n]
                inst = tables.instance.get(a.get("instance_token", ""), {})
                category = tables.category.get(inst.get("category_token", ""), {}).get(
                    "name", "object_fallback")

                q_box_ego = quat_multiply(q_ego_inv, a["rotation"])
                q_box_sensor = quat_multiply(q_cs_inv, q_box_ego)
                center_ego = apply_pose(global_to_ego, centers_g[n])[0]
                center_sensor = apply_pose(global_to_sensor, centers_g[n])[0]
                rot_box_ego = global_to_ego[:3, :3] @ rot_g[n]
                size_wlh = _vec(sizes[n], 4)

                vis_token = a.get("visibility_token")
                attrs: dict = {}
                if vis_token not in (None, ""):
                    attrs["nuscenes_visibility"] = tables.visibility.get(str(vis_token), {}).get(
                        "level", str(vis_token))
                attr_names = [tables.attribute[t]["name"] for t in a.get("attribute_tokens") or []
                              if t in tables.attribute]
                if attr_names:
                    attrs["nuscenes_attributes"] = attr_names

                provenance = {
                    CUBOID_KEY: {"center": _vec(center_ego, 4), "size": size_wlh,
                                 "yaw": round(heading_of(rot_box_ego), 6)},
                    "nuscenes_box": {
                        "size_wlh": size_wlh,
                        "global": {"center": _vec(centers_g[n], 4),
                                   "rotation": _vec(quat_canonical(a["rotation"]), 8)},
                        "ego": {"center": _vec(center_ego, 4), "rotation": _vec(quat_canonical(q_box_ego), 8)},
                        "sensor": {"channel": channel, "center": _vec(center_sensor, 4),
                                   "rotation": _vec(quat_canonical(q_box_sensor), 8)},
                    },
                    "nuscenes": {
                        "annotation_token": a.get("token"), "instance_token": a.get("instance_token"),
                        "sample_token": stok, "sample_data_token": sd.get("token"),
                        "scene_token": sample.get("scene_token"), "scene_name": scene.get("name"),
                        "channel": channel, "capture_timestamp_us": sd.get("timestamp"),
                        "num_lidar_pts": a.get("num_lidar_pts"), "num_radar_pts": a.get("num_radar_pts"),
                    },
                }
                frame.objects.append(ImportObject(
                    name=category, bbox=bbox, attrs=attrs, track_ref=a.get("instance_token"),
                    provenance=provenance,
                ))
                n_objects += 1

    # Time order, then channel: a stable listing, since services.imports.run fingerprints it to resume.
    frames.sort(key=lambda f: (f.ts_ns or 0, f.cam_id, f.image_ref))
    log.info("import_nuscenes.standard", table_dir=str(tables.table_dir), frames=len(frames),
             objects=n_objects, annotations=len(tables.sample_annotation), unusable_boxes=n_bad_boxes,
             frames_without_calibration=n_uncalibrated, dangling_sample_data=dangling)
    return frames


def parse(root: Path, frame_time: str = "sample") -> list[ImportFrame]:
    """Parse a nuScenes table directory under root: a standard release, or LabeloxAV's own export.

    `frame_time` applies to a standard release only: "sample" stamps every camera of a sample with the
    sample's timestamp (one rig instant), "capture" with each camera's own capture time.
    """
    root = Path(root)
    sd_path = find_file(root, "sample_data.json")
    ann_path = None
    if sd_path is not None:
        beside = sd_path.parent / "sample_annotation.json"
        ann_path = beside if beside.is_file() else find_file(root, "sample_annotation.json")
    if sd_path is None or ann_path is None:
        raise FileNotFoundError("no nuScenes sample_data.json / sample_annotation.json found")

    annotations = load_json(ann_path)
    if not isinstance(annotations, list):
        raise ValueError(f"{ann_path.name} is not a nuScenes table (expected a list of rows)")
    if _is_labelox_export(annotations, sd_path.parent):
        sample_data = load_json(sd_path)
        if not isinstance(sample_data, list):
            raise ValueError(f"{sd_path.name} is not a nuScenes table (expected a list of rows)")
        frames = _parse_labelox_export(sample_data, annotations)
        log.info("import_nuscenes.parsed", frames=len(frames))
        return frames
    del annotations                                    # load_tables reads the table again, indexed
    return _parse_standard(root, load_tables(root, table_dir=sd_path.parent), frame_time)
