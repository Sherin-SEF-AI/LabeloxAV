"""ClassificationTask: train on the crops the corpus already holds, and on the scene tags it already carries.

Two things the system knew and could not learn from.

The first is fine-grained class confusion. The detector proposes `autorickshaw` and the reviewer corrects it
to `delivery_rider_bike`; that correction becomes a better box label and nothing more. A crop classifier
trained on the corrected crops attacks exactly the confusions the confusion matrix keeps reporting, and it
does so at a fraction of a detector's cost because a crop is small and there is no localisation to learn.

The second is scene state. `Frame.scene` has carried weather, time of day, road type and density since
migration 0062, populated by the scene model and corrected by reviewers, and no model was ever trained to
predict it. Every slice metric in VERDYX is computed over those axes, so a frame that arrives without them
cannot be sliced at all.

The builder therefore has two sources, chosen by `source`:

- `object`: one image per object crop, foldered by class name.
- `frame`: one image per frame, foldered by the value of a single `scene` axis.

Both write the folder-per-class layout ultralytics classification expects, which is why no data.yaml lists
names: the directory names *are* the classes, and inventing a yaml alongside them would create a second
source of truth that can disagree.
"""

from __future__ import annotations

import dataclasses
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.dataset_builder import _exclude_gold_objects
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_classification")

# The scene axes the frame model populates. Pinned rather than discovered, so a typo in one row cannot
# invent a class with a single example in it.
SCENE_AXES = ("weather", "time_of_day", "road_type", "density")


@dataclass
class ClassificationBuildSpec:
    name: str = "cls-v1"
    source: str = "object"            # object (crops by class) | frame (whole frame by a scene axis)
    scene_axis: str = "weather"       # only when source == "frame"
    conf_floor: float = 0.2
    # Crops are cheap, so the ceiling is higher than the detector's. It still exists: without it one common
    # class swamps the rest and the model learns the prior instead of the appearance.
    max_per_class: int = 2000
    min_per_class: int = 25
    val_frac: float = 0.2
    states: list[str] = field(default_factory=lambda: ["accepted", "approved"])
    classes: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    seed: int = 7
    exclude_gold_id: str | None = None
    # A crop smaller than this carries no appearance information at any sensible input size; including it
    # teaches the model to classify noise.
    min_crop_px: int = 24
    pad: float = 0.10
    # Grouping by session matters more here than anywhere: consecutive dashcam crops of the same vehicle are
    # near-duplicates, and splitting them per crop would put the same object on both sides of the split.
    group_split_by_session: bool = True


async def _select_objects(spec: ClassificationBuildSpec) -> list[dict]:
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession

    onto = get_ontology()
    wanted = {onto.by_name(n).id for n in spec.classes} if spec.classes else None

    async with get_sessionmaker()() as db:
        stmt = (
            select(Object, Frame.img_uri, Frame.width, Frame.height, Frame.session_id, Frame.frame_id)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(Object.state != "rejected", Object.conf >= spec.conf_floor)
        )
        if spec.states:
            stmt = stmt.where(Object.state.in_(spec.states))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        if wanted:
            stmt = stmt.where(Object.class_id.in_(sorted(wanted)))
        rows = (await db.execute(stmt)).all()

    out: list[dict] = []
    for obj, img_uri, w, h, session_id, frame_id in rows:
        x1, y1, x2, y2 = (float(v) for v in obj.bbox)
        if (x2 - x1) < spec.min_crop_px or (y2 - y1) < spec.min_crop_px:
            continue
        out.append({
            "object_id": str(obj.object_id), "frame_id": str(frame_id), "session_id": str(session_id),
            "img_uri": img_uri, "w": w, "h": h, "bbox": [x1, y1, x2, y2],
            "label": onto.by_id(obj.class_id).name,
            "gold": obj.source == "human" and obj.state == "accepted",
        })
    return out


async def _select_frames(spec: ClassificationBuildSpec) -> list[dict]:
    from sqlalchemy import select

    from db.models import Frame
    from db.models import Session as DbSession

    if spec.scene_axis not in SCENE_AXES:
        raise ValueError(f"scene_axis must be one of {SCENE_AXES}")

    async with get_sessionmaker()() as db:
        stmt = (select(Frame, DbSession.city)
                .join(DbSession, Frame.session_id == DbSession.session_id)
                .where(Frame.scene.isnot(None)))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        rows = (await db.execute(stmt)).all()

    out: list[dict] = []
    for frame, _city in rows:
        value = (frame.scene or {}).get(spec.scene_axis)
        if not value or not isinstance(value, str):
            continue
        out.append({
            "object_id": None, "frame_id": str(frame.frame_id), "session_id": str(frame.session_id),
            "img_uri": frame.img_uri, "w": frame.width, "h": frame.height, "bbox": None,
            "label": value.strip().lower().replace(" ", "_"), "gold": False,
        })
    return out


def _balance(cand: list[dict], spec: ClassificationBuildSpec) -> tuple[list[dict], dict]:
    """Cap each class and drop the ones too thin to learn, reporting both rather than doing it silently.

    A class with three examples does not produce a classifier for that class; it produces a model that has
    memorised three images and reports a confident, meaningless score on everything else.
    """
    rng = random.Random(spec.seed)
    by_label: dict[str, list[dict]] = {}
    for c in cand:
        by_label.setdefault(c["label"], []).append(c)

    kept: list[dict] = []
    dropped: dict[str, int] = {}
    capped: dict[str, int] = {}
    for label, items in by_label.items():
        if len(items) < spec.min_per_class:
            dropped[label] = len(items)
            continue
        if len(items) > spec.max_per_class:
            capped[label] = len(items) - spec.max_per_class
            rng.shuffle(items)
            items = items[:spec.max_per_class]
        kept.extend(items)
    return kept, {"dropped_thin_classes": dropped, "capped_classes": capped}


def _split_sessions(cand: list[dict], val_frac: float, seed: int,
                    group_by_session: bool) -> set[str]:
    """Which session ids go to validation.

    Grouping by session is not a nicety here. Consecutive dashcam crops of one vehicle are near-identical,
    so a per-crop split puts the same object on both sides and the validation number measures memorisation.
    """
    if not group_by_session:
        return set()
    sessions = sorted({c["session_id"] for c in cand})
    if len(sessions) < 2:
        return set()
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n = max(1, int(len(sessions) * val_frac))
    return set(sessions[:n])


async def build_classification_dataset(spec: ClassificationBuildSpec) -> dict:
    settings = get_settings()
    store = get_object_store()

    out = settings.scratch_path() / "training" / spec.name
    if out.exists():
        shutil.rmtree(out)

    cand = (await _select_objects(spec) if spec.source == "object"
            else await _select_frames(spec))
    if spec.source == "object":
        cand, excluded_gold = await _exclude_gold_objects(cand, spec.exclude_gold_id)
    else:
        excluded_gold = 0
    if not cand:
        raise ValueError(
            "no candidates: a crop classifier needs reviewed objects, and a scene classifier needs frames "
            "whose scene axis has been populated")

    cand, balance = _balance(cand, spec)
    if not cand:
        raise ValueError(
            f"every class had fewer than {spec.min_per_class} examples; "
            f"counts were {balance['dropped_thin_classes']}")

    val_sessions = _split_sessions(cand, spec.val_frac, spec.seed, spec.group_split_by_session)
    labels = sorted({c["label"] for c in cand})
    for split in ("train", "val"):
        for label in labels:
            (out / split / label).mkdir(parents=True, exist_ok=True)

    # Group by frame so each source image is fetched and decoded once even when it yields many crops.
    by_frame: dict[str, list[dict]] = {}
    for c in cand:
        by_frame.setdefault(c["frame_id"], []).append(c)

    counts = {"train": 0, "val": 0}
    per_class: dict[str, int] = {}
    unreadable = 0
    for fid, items in by_frame.items():
        try:
            buf = np.frombuffer(store.get_bytes(items[0]["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            unreadable += 1
            continue
        split = "val" if items[0]["session_id"] in val_sessions else "train"
        for c in items:
            crop = _crop(img, c["bbox"], spec.pad) if c["bbox"] else img
            if crop is None or crop.size == 0:
                continue
            stem = c["object_id"] or fid
            if not cv2.imwrite(str(out / split / c["label"] / f"{stem}.jpg"), crop):
                continue
            counts[split] += 1
            per_class[c["label"]] = per_class.get(c["label"], 0) + 1

    if counts["train"] == 0:
        raise ValueError("no images could be written; every source image was unreadable")

    result = {
        "name": spec.name, "dir": str(out),
        # The directory IS the dataset for a classification run, so this is the path ultralytics is given.
        "data_yaml": str(out),
        "classes": len(labels), "class_names": labels,
        "n_train_images": counts["train"], "n_val_images": counts["val"],
        "per_class": per_class, "unreadable_frames": unreadable,
        "source": spec.source,
        "scene_axis": spec.scene_axis if spec.source == "frame" else None,
        "split": "session_grouped" if spec.group_split_by_session else "per_item",
        "excluded_gold_id": spec.exclude_gold_id, "excluded_gold_objects": excluded_gold,
        "ontology_version": get_ontology().version,
        "task": "classification",
        **balance,
    }
    log.info("clsset.built", classes=len(labels), train=counts["train"], val=counts["val"],
             dropped=len(balance["dropped_thin_classes"]))
    return result


def _crop(img, bbox: list[float], pad: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


_SPEC_FIELDS = {f.name for f in dataclasses.fields(ClassificationBuildSpec)}


class ClassificationTask:
    task_type = "classification"

    def default_base_weights(self) -> str:
        return get_settings().training.classification_weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _SPEC_FIELDS}
        spec["name"] = cfg["name"]
        return await build_classification_dataset(ClassificationBuildSpec(**spec))

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        from ultralytics import YOLO

        settings = get_settings()
        name = hparams["name"]
        epochs = int(hparams.get("epochs", settings.training.default_epochs))
        # Classification runs at a smaller input than detection: a crop has no small distant objects to
        # resolve, and 224 is what the pretrained classification heads expect.
        imgsz = int(hparams.get("imgsz", 224))
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
                          "metrics": {"top1": round(float(m.get("metrics/accuracy_top1", 0.0)), 4),
                                      "top5": round(float(m.get("metrics/accuracy_top5", 0.0)), 4)}})
                if should_stop and should_stop():
                    trainer.stop = True
            except Exception:  # noqa: BLE001
                pass

        model.add_callback("on_fit_epoch_end", _on_epoch)
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
                    project=project, name=name, exist_ok=True, verbose=False, plots=False,
                    batch=batch, patience=max(5, epochs // 3), workers=8, seed=7,
                    resume=bool(hparams.get("resume")))
        return str(Path(project) / name / "weights" / "best.pt")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        from ultralytics import YOLO

        res = YOLO(weights).val(data=data_yaml, imgsz=min(imgsz, 224), verbose=False)
        top1 = float(getattr(getattr(res, "top1", None) or 0, "item", lambda: getattr(res, "top1", 0.0))())
        top5 = float(getattr(getattr(res, "top5", None) or 0, "item", lambda: getattr(res, "top5", 0.0))())
        # map50/map are carried so the shared gate and the job row read one vocabulary across every task,
        # not because a classifier has an mAP: top-1 accuracy is the number, under the shared name.
        return {"top1": top1, "top5": top5, "map50": top1, "map": top1}

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        from services.training import eval as eval_mod

        crit = {k: v for k, v in (criteria or {}).items() if k in ("min_map_delta", "max_class_drop")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(ClassificationTask())
