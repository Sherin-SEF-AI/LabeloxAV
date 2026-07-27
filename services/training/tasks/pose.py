"""PoseTask: train a keypoint model on the skeletons the corpus already carries.

`Object.keypoints` has been a first-class annotation field all along (the editor has a pose tool, the API
round-trips it, the exporters emit it), and no model could ever be trained on it. For a VRU-heavy ODD that
is a real gap: body pose is the signal that separates a pedestrian about to step off a kerb from one waiting,
and intent prediction had no access to it.

Keypoints are stored as {"skeleton": name, "points": [[x, y, visibility], ...]} in image pixels. YOLO-pose
wants one line per instance: class, normalized box, then normalized x y v per keypoint, with a fixed count
per skeleton. A dataset therefore covers exactly one skeleton; mixing two would misalign the indices, so the
builder refuses rather than silently interleaving them.
"""

from __future__ import annotations

import dataclasses
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.dataset_builder import _exclude_gold_objects, _split_val_frames
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_pose")

# The skeletons the editor offers. The count is part of the label format, so it is pinned here rather than
# inferred from whatever the first annotation happens to contain.
SKELETONS = {"person_17": 17}


@dataclass
class PoseBuildSpec:
    name: str = "pose-v1"
    skeleton: str = "person_17"
    conf_floor: float = 0.2
    max_per_class: int = 400
    val_frac: float = 0.2
    states: list[str] = field(default_factory=list)
    seed: int = 7
    cities: list[str] = field(default_factory=list)
    group_split_by_session: bool = True
    exclude_gold_id: str | None = None
    # An instance whose keypoints are almost all marked invisible carries no pose signal and mostly teaches
    # the model to predict absence, so it is not a candidate.
    min_visible_keypoints: int = 3


def keypoint_label_line(class_index: int, bbox: list[float], points: list[list[float]],
                        width: int, height: int) -> str | None:
    """One YOLO-pose label line: class, normalized cx cy w h, then x y v per keypoint."""
    x1, y1, x2, y2 = bbox
    w, h = max(width, 1), max(height, 1)
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    parts = [f"{class_index}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for p in points:
        px, py = float(p[0]) / w, float(p[1]) / h
        vis = int(p[2]) if len(p) > 2 else 2
        # An invisible keypoint is written at the origin with v=0, which is the convention the loss uses to
        # skip it. Writing its stale coordinates instead would train the model toward a point nobody marked.
        if vis == 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [f"{min(max(px, 0.0), 1.0):.6f}", f"{min(max(py, 0.0), 1.0):.6f}", str(vis)]
    return " ".join(parts)


async def _select(spec: PoseBuildSpec) -> list[dict]:
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession

    n_points = SKELETONS[spec.skeleton]
    async with get_sessionmaker()() as db:
        stmt = (
            select(Object, Frame.frame_id, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(Object.state != "rejected", Object.conf >= spec.conf_floor,
                   Object.keypoints.isnot(None))
        )
        if spec.states:
            stmt = stmt.where(Object.state.in_(spec.states))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        rows = (await db.execute(stmt)).all()

    cand: list[dict] = []
    for obj, frame_id, img_uri, w, h, session_id in rows:
        kp = obj.keypoints or {}
        if kp.get("skeleton") != spec.skeleton:
            continue
        points = kp.get("points") or []
        if len(points) != n_points:
            continue     # a different point count would misalign every index in the label
        if sum(1 for p in points if len(p) > 2 and int(p[2]) > 0) < spec.min_visible_keypoints:
            continue
        cand.append({
            "frame_id": str(frame_id), "session_id": str(session_id), "object_id": str(obj.object_id),
            "img_uri": img_uri, "w": w, "h": h, "class_id": obj.class_id,
            "bbox": list(obj.bbox), "points": points,
            "gold": obj.source == "human" and obj.state == "accepted",
        })
    return cand


async def build_pose_dataset(spec: PoseBuildSpec) -> dict:
    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    if spec.skeleton not in SKELETONS:
        raise ValueError(f"unknown skeleton {spec.skeleton!r}; known: {sorted(SKELETONS)}")

    out = settings.scratch_path() / "training" / spec.name
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cand = await _select(spec)
    cand, excluded_gold = await _exclude_gold_objects(cand, spec.exclude_gold_id)
    if not cand:
        raise ValueError(f"no objects carry a {spec.skeleton} skeleton; pose training needs keypoint labels")

    present = sorted({c["class_id"] for c in cand})
    idx_of = {cid: i for i, cid in enumerate(present)}
    names = {i: onto.by_id(cid).name for cid, i in idx_of.items()}

    by_frame: dict[str, list[dict]] = {}
    gold_frames: set[str] = set()
    for c in cand:
        by_frame.setdefault(c["frame_id"], []).append(c)
        if c["gold"]:
            gold_frames.add(c["frame_id"])

    n_val = max(1, int(len(by_frame) * spec.val_frac)) if len(by_frame) > 4 else 0
    val_set = _split_val_frames(by_frame, gold_frames, n_val, random.Random(spec.seed),
                                group_by_session=spec.group_split_by_session)

    n_train_img = n_val_img = n_train_obj = n_val_obj = 0
    last_uri, last_img = None, None
    for fid, objs in by_frame.items():
        split = "val" if fid in val_set else "train"
        first = objs[0]
        if first["img_uri"] != last_uri:
            try:
                buf = np.frombuffer(store.get_bytes(first["img_uri"]), dtype=np.uint8)
                last_uri, last_img = first["img_uri"], cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                last_uri, last_img = first["img_uri"], None
        if last_img is None:
            continue
        lines = [ln for o in objs
                 if (ln := keypoint_label_line(idx_of[o["class_id"]], o["bbox"], o["points"], o["w"], o["h"]))]
        if not lines:
            continue
        cv2.imwrite(str(out / f"images/{split}/{fid}.jpg"), last_img)
        (out / f"labels/{split}/{fid}.txt").write_text("\n".join(lines) + "\n")
        if split == "train":
            n_train_img += 1
            n_train_obj += len(lines)
        else:
            n_val_img += 1
            n_val_obj += len(lines)

    n_points = SKELETONS[spec.skeleton]
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": names,
        # kpt_shape is how ultralytics knows the label layout; a wrong value silently misreads every line.
        "kpt_shape": [n_points, 3],
    }, sort_keys=False))

    result = {
        "name": spec.name, "dir": str(out), "data_yaml": str(data_yaml),
        "classes": len(names), "n_train_images": n_train_img, "n_val_images": n_val_img,
        "n_train_objects": n_train_obj, "n_val_objects": n_val_obj,
        "gold_frames": len(gold_frames), "ontology_version": onto.version,
        "split": "session_grouped" if spec.group_split_by_session else "per_frame",
        "excluded_gold_id": spec.exclude_gold_id, "excluded_gold_objects": excluded_gold,
        "task": "pose", "skeleton": spec.skeleton, "keypoints": n_points,
    }
    log.info("poseset.built", **{k: result[k] for k in ("classes", "n_train_images", "n_val_images")})
    return result


_SPEC_FIELDS = {f.name for f in dataclasses.fields(PoseBuildSpec)}


class PoseTask:
    task_type = "pose"

    def default_base_weights(self) -> str:
        return get_settings().training.pose_weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _SPEC_FIELDS}
        spec["name"] = cfg["name"]
        return await build_pose_dataset(PoseBuildSpec(**spec))

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        from ultralytics import YOLO

        settings = get_settings()
        name = hparams["name"]
        epochs = int(hparams.get("epochs", settings.training.default_epochs))
        imgsz = int(hparams.get("imgsz", settings.training.default_imgsz))
        batch = int(hparams.get("batch", settings.training.default_batch))
        project = str(settings.scratch_path() / "training" / "runs")
        should_stop = hparams.get("_should_stop")

        model = YOLO(base_weights)

        def _on_epoch(trainer):
            try:
                ep = int(getattr(trainer, "epoch", 0)) + 1
                m = getattr(trainer, "metrics", {}) or {}
                progress({"stage": "train", "epoch": ep,
                          "total_epochs": int(getattr(trainer, "epochs", epochs)),
                          "metrics": {"map50_pose": round(float(m.get("metrics/mAP50(P)", 0.0)), 4),
                                      "map50_box": round(float(m.get("metrics/mAP50(B)", 0.0)), 4)}})
                if should_stop and should_stop():
                    trainer.stop = True
            except Exception:  # noqa: BLE001
                pass

        model.add_callback("on_fit_epoch_end", _on_epoch)
        model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
            project=project, name=name, exist_ok=True, verbose=False, plots=False,
            batch=batch, patience=max(5, epochs // 3), workers=8, seed=7,
            resume=bool(hparams.get("resume")),
        )
        return str(Path(project) / name / "weights" / "best.pt")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        from ultralytics import YOLO

        res = YOLO(weights).val(data=data_yaml, imgsz=imgsz, verbose=False)
        pose = getattr(res, "pose", None)
        box = getattr(res, "box", None)
        out = {
            "map50_pose": float(getattr(pose, "map50", 0.0)) if pose else None,
            "map_pose": float(getattr(pose, "map", 0.0)) if pose else None,
            "map50_box": float(getattr(box, "map50", 0.0)) if box else None,
        }
        # Gate on the keypoint metric: a pose model whose boxes improve while its joints degrade has
        # regressed at the task it was trained for.
        out["map50"] = out["map50_pose"]
        out["map"] = out["map_pose"]
        return out

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        from services.training import eval as eval_mod

        crit = {k: v for k, v in (criteria or {}).items() if k in ("min_map_delta", "max_class_drop")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(PoseTask())
