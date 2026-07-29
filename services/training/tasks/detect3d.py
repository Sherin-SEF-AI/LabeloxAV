"""Detection3dTask: build a KITTI-format 3D dataset from the cuboids the corpus holds, and refuse to train.

This one is deliberately half a plugin, and the half it is missing is stated rather than faked.

The dataset builder is real and complete. `Object3D` cuboids have been annotated in the 3D editor, lifted
from 2D, linked back to camera boxes, and tracked, and none of that could ever leave the database in a form
a 3D detector reads. That is the part this file fixes: it writes the KITTI label/calib/velodyne layout that
OpenPCDet, MMDetection3D and Pointcept all consume, so the corpus is trainable the moment a runtime exists.

`train` raises. There is no OpenPCDet or Pointcept in this environment and no GPU pod to run one on, and a
plugin that quietly trained a 2D model on 3D labels and reported an mAP would be worse than one that
refuses: the number would look like progress. The refusal names exactly what is missing.

There is a second, physical caveat the refusal also carries. The running pipeline lifts cuboids from
monocular depth rather than from a real LiDAR sweep, so their absolute scale inherits the depth model's
error. Training a detector on those without real sweeps teaches it that error as ground truth, which is a
worse outcome than not training at all.
"""

from __future__ import annotations

import dataclasses
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_detect3d")


class NativeTrainingUnavailable(RuntimeError):
    """Raised instead of training a 3D detector that this environment cannot train."""


@dataclass
class Detect3dBuildSpec:
    name: str = "det3d-v1"
    conf_floor: float = 0.2
    states: list[str] = field(default_factory=lambda: ["accepted", "approved"])
    classes: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    val_frac: float = 0.2
    max_frames: int = 8000
    seed: int = 7
    group_split_by_session: bool = True
    # Whether to require a real point cloud. On by default: a cuboid lifted from monocular depth carries
    # that model's scale error, and training on it teaches the error as truth.
    require_point_cloud: bool = True
    min_dims_m: float = 0.2


def kitti_label_line(class_name: str, center: list[float], size: list[float], yaw: float,
                     bbox_2d: list[float] | None = None, truncated: float = 0.0,
                     occluded: int = 0) -> str:
    """One KITTI 3D label line.

    KITTI's own conventions, not a convenient reinterpretation of them, because every reader assumes them:
    dimensions are height width length in that order, the location is the BOTTOM centre rather than the
    centroid, and rotation_y is measured about the camera's downward axis.
    """
    width, length, height = (float(size[0]), float(size[1]), float(size[2]))
    x, y, z = (float(center[0]), float(center[1]), float(center[2]))
    bx = bbox_2d or [0.0, 0.0, 0.0, 0.0]
    alpha = _alpha_from_yaw(yaw, x, z)
    return (f"{class_name} {truncated:.2f} {int(occluded)} {alpha:.2f} "
            f"{bx[0]:.2f} {bx[1]:.2f} {bx[2]:.2f} {bx[3]:.2f} "
            f"{height:.2f} {width:.2f} {length:.2f} "
            f"{x:.2f} {y + height / 2:.2f} {z:.2f} {float(yaw):.2f}")


def _alpha_from_yaw(yaw: float, x: float, z: float) -> float:
    """Observation angle: the yaw with the viewing direction removed.

    KITTI carries both because a detector sees an object's appearance from its own viewpoint, and a box at
    the edge of the image with the same world yaw looks quite different from one straight ahead.
    """
    alpha = float(yaw) - math.atan2(float(x), max(float(z), 1e-6))
    while alpha > math.pi:
        alpha -= 2 * math.pi
    while alpha < -math.pi:
        alpha += 2 * math.pi
    return alpha


def kitti_calib_text(intrinsics: dict | None) -> str:
    """The calib file. Written from the session's real intrinsics when present.

    When they are absent the identity projection is written and the dataset records that it did, rather than
    inventing a plausible-looking focal length: a wrong calibration silently ruins every 3D box that is
    later projected through it, and a reader has no way to tell.
    """
    fx = float((intrinsics or {}).get("fx") or 0.0)
    fy = float((intrinsics or {}).get("fy") or 0.0)
    cx = float((intrinsics or {}).get("cx") or 0.0)
    cy = float((intrinsics or {}).get("cy") or 0.0)
    if fx <= 0 or fy <= 0:
        p2 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    else:
        p2 = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rows = [f"P{i}: " + " ".join(f"{v:.6e}" for v in p2) for i in range(4)]
    rows.append("R0_rect: " + " ".join(f"{v:.6e}" for v in ident))
    rows.append("Tr_velo_to_cam: " + " ".join(f"{v:.6e}" for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
    rows.append("Tr_imu_to_velo: " + " ".join(f"{v:.6e}" for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))
    return "\n".join(rows) + "\n"


async def _select(spec: Detect3dBuildSpec) -> dict[str, dict]:
    from sqlalchemy import select

    from db.models import Frame, Object3D
    from db.models import Session as DbSession

    onto = get_ontology()
    wanted = {onto.by_name(n).id for n in spec.classes} if spec.classes else None

    frames: dict[str, dict] = {}
    async with get_sessionmaker()() as db:
        stmt = (select(Object3D, Frame.frame_id, Frame.img_uri, Frame.session_id,
                       Frame.width, Frame.height, DbSession.sensors)
                .join(Frame, Object3D.frame_id == Frame.frame_id)
                .join(DbSession, Frame.session_id == DbSession.session_id)
                .where(Object3D.state != "rejected"))
        if spec.states:
            stmt = stmt.where(Object3D.state.in_(spec.states))
        if wanted:
            stmt = stmt.where(Object3D.class_id.in_(sorted(wanted)))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        rows = (await db.execute(stmt)).all()

        for obj, frame_id, img_uri, session_id, w, h, sensors in rows:
            dims = list(obj.dims or [])
            if len(dims) != 3 or min(dims) < spec.min_dims_m:
                continue
            f = frames.setdefault(str(frame_id), {
                "img_uri": img_uri, "session_id": str(session_id), "w": w, "h": h,
                "intrinsics": (sensors or {}).get("intrinsics"), "objects": [], "cloud_uri": None})
            f["objects"].append({
                "class_name": onto.by_id(obj.class_id).name,
                "center": list(obj.center or [0, 0, 0]), "dims": dims,
                "yaw": float(obj.yaw or 0.0),
            })

        if spec.require_point_cloud:
            from db.models import LidarCloud

            fids = [f for f in frames]
            if fids:
                import uuid as _uuid

                clouds = (await db.execute(
                    select(LidarCloud).where(
                        LidarCloud.frame_id.in_([_uuid.UUID(f) for f in fids])))).scalars().all()
                for c in clouds:
                    if str(c.frame_id) in frames:
                        frames[str(c.frame_id)]["cloud_uri"] = c.points_uri
    return frames


async def build_detect3d_dataset(spec: Detect3dBuildSpec) -> dict:
    """Write the KITTI 3D layout. This part works and is what makes the corpus trainable elsewhere."""
    settings = get_settings()
    store = get_object_store()

    out = settings.scratch_path() / "training" / spec.name
    if out.exists():
        shutil.rmtree(out)
    for sub in ("training/label_2", "training/calib", "training/image_2", "training/velodyne",
                "ImageSets"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    frames = await _select(spec)
    if not frames:
        raise ValueError("no 3D cuboids match; a 3D detector needs annotated Object3D rows")

    if spec.require_point_cloud:
        without = [f for f, v in frames.items() if not v["cloud_uri"]]
        frames = {f: v for f, v in frames.items() if v["cloud_uri"]}
        if not frames:
            raise ValueError(
                f"{len(without)} frames carry cuboids and none has a point cloud. Those cuboids were lifted "
                "from monocular depth, so their absolute scale carries the depth model's error; training on "
                "them teaches that error as ground truth. Set require_point_cloud=false to build anyway.")

    rng = random.Random(spec.seed)
    fids = sorted(frames)
    if len(fids) > spec.max_frames:
        rng.shuffle(fids)
        fids = fids[:spec.max_frames]

    val_sessions: set[str] = set()
    if spec.group_split_by_session:
        sessions = sorted({frames[f]["session_id"] for f in fids})
        if len(sessions) > 1:
            rng.shuffle(sessions)
            val_sessions = set(sessions[:max(1, int(len(sessions) * spec.val_frac))])

    train_ids: list[str] = []
    val_ids: list[str] = []
    n_objects = 0
    no_calib = 0
    for i, fid in enumerate(sorted(fids)):
        f = frames[fid]
        stem = f"{i:06d}"
        lines = [kitti_label_line(o["class_name"], o["center"], o["dims"], o["yaw"])
                 for o in f["objects"]]
        if not lines:
            continue
        (out / "training/label_2" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        if not (f["intrinsics"] or {}).get("fx"):
            no_calib += 1
        (out / "training/calib" / f"{stem}.txt").write_text(kitti_calib_text(f["intrinsics"]))

        try:
            (out / "training/image_2" / f"{stem}.png").write_bytes(store.get_bytes(f["img_uri"]))
        except Exception:  # noqa: BLE001 - a missing image does not invalidate the 3D label
            pass
        if f["cloud_uri"]:
            _write_velodyne(store, f["cloud_uri"], out / "training/velodyne" / f"{stem}.bin")

        (val_ids if f["session_id"] in val_sessions else train_ids).append(stem)
        n_objects += len(lines)

    (out / "ImageSets/train.txt").write_text("\n".join(train_ids) + "\n")
    (out / "ImageSets/val.txt").write_text("\n".join(val_ids) + "\n")

    result = {
        "name": spec.name, "dir": str(out), "data_yaml": str(out),
        "format": "kitti3d",
        "classes": len({o["class_name"] for f in frames.values() for o in f["objects"]}),
        "n_train_images": len(train_ids), "n_val_images": len(val_ids),
        "n_objects": n_objects,
        # Surfaced rather than hidden: a reader has no way to tell an identity projection from a real one,
        # and every box projected through it would be silently wrong.
        "frames_without_calibration": no_calib,
        "point_clouds_required": spec.require_point_cloud,
        "split": "session_grouped" if spec.group_split_by_session else "per_frame",
        "task": "detect3d",
        "ontology_version": get_ontology().version,
    }
    log.info("det3dset.built", train=len(train_ids), val=len(val_ids), objects=n_objects,
             no_calib=no_calib)
    return result


def _write_velodyne(store, uri: str, dest: Path) -> None:
    """Write the cloud as KITTI's float32 x y z intensity binary, which is what every reader expects."""
    try:
        raw = store.get_bytes(uri)
    except Exception:  # noqa: BLE001
        return
    try:
        arr = np.load(__import__("io").BytesIO(raw), allow_pickle=False)
        pts = np.asarray(arr, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return
        if pts.shape[1] == 3:
            # No intensity channel in the source. Zeros, not a synthesised value: a reader treats intensity
            # as a measurement, and inventing one is inventing sensor data.
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        pts[:, :4].astype(np.float32).tofile(dest)
    except Exception:  # noqa: BLE001
        dest.write_bytes(raw)   # already binary in some other layout: pass it through unchanged


_SPEC_FIELDS = {f.name for f in dataclasses.fields(Detect3dBuildSpec)}


class Detection3dTask:
    task_type = "detect3d"

    def default_base_weights(self) -> str:
        return get_settings().training.detect3d_weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _SPEC_FIELDS}
        spec["name"] = cfg["name"]
        return await build_detect3d_dataset(Detect3dBuildSpec(**spec))

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        raise NativeTrainingUnavailable(
            "3D detection training needs OpenPCDet or MMDetection3D and a CUDA device, neither of which is "
            "present here. The KITTI-format dataset this task builds is complete and can be trained "
            f"elsewhere: point a 3D detector at {data_yaml}. Training a 2D model on these labels and "
            "reporting an mAP would look like progress and would not be.")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        raise NativeTrainingUnavailable(
            "3D evaluation needs the detector above. The 3D and BEV AP kernels exist in core/accel/ap3d.py "
            "and are tested; what is missing is a model to score, not a metric to score it with.")

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        from services.training import eval as eval_mod

        crit = {k: v for k, v in (criteria or {}).items() if k in ("min_map_delta", "max_class_drop")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(Detection3dTask())
