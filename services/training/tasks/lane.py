"""LaneTask: train a lane model on the lane lines and drivable surfaces the corpus already holds.

`Lane` rows have existed since M2.1 and `DrivableMask` since M2.2. Both are annotated in the editor, both are
corrected by reviewers, and neither could ever improve a model: the lane proposer is a classical Hough-style
fallback and the pod-side learned proposer raises rather than pretending. So the corpus accumulated exactly
the supervision a lane model needs and had no way to consume it.

The representation is the design decision. Lane lines are stored as control points, which is right for
editing and wrong for training: a polyline has no width, so an IoU against it is degenerate. This builds a
segmentation dataset instead, rasterising each line to a fixed-width ribbon and each drivable surface to its
stored mask, then trains the segmentation head. That is what every published lane model does for the same
reason, and it means the lane task inherits the mask metrics that already exist rather than needing its own.

Two sources, combined or separate:

- `lane`: one class per lane type (solid, dashed, road_edge, ...), rasterised at `line_width_px`.
- `drivable`: the ternary surface mask, whose classes are already regions rather than lines.
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
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_lane")

# Lane types the editor writes. Fixed here because the class index is part of every label line, so
# discovering them from data would renumber the classes whenever the corpus changed.
LANE_TYPES = ("solid", "dashed", "double", "road_edge", "implicit", "fallback")
DRIVABLE_CLASSES = ("drivable", "non_drivable", "fallback")


@dataclass
class LaneBuildSpec:
    name: str = "lane-v1"
    sources: list[str] = field(default_factory=lambda: ["lane"])   # lane | drivable
    lane_types: list[str] = field(default_factory=list)            # empty means every type present
    # A lane line has no width. This is the ribbon it is drawn as, and it is a real modelling choice: too
    # thin and the positive class is a handful of pixels the loss ignores, too thick and the model learns a
    # blob whose edges mean nothing.
    line_width_px: int = 12
    min_points: int = 2
    val_frac: float = 0.2
    sources_states: list[str] = field(default_factory=lambda: ["human", "propagated", "proposed"])
    cities: list[str] = field(default_factory=list)
    max_frames: int = 4000
    seed: int = 7
    # Same reason as everywhere else in this codebase: consecutive dashcam frames are near-duplicates, and a
    # per-frame split measures memorisation rather than generalisation.
    group_split_by_session: bool = True


def rasterize_lane(points: list[list[float]], width: int, height: int,
                   line_width_px: int) -> np.ndarray | None:
    """Draw one lane polyline as a filled ribbon mask.

    Returns None when the line is degenerate (fewer than two distinct points), rather than an empty mask
    that would be written out as a label containing nothing.
    """
    pts = [(int(round(p[0])), int(round(p[1]))) for p in points if len(p) >= 2]
    pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    if len(pts) < 2:
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], isClosed=False, color=255,
                  thickness=max(1, line_width_px), lineType=cv2.LINE_AA)
    return canvas


def mask_to_polygons(mask: np.ndarray, width: int, height: int,
                     min_area_px: int = 24) -> list[list[float]]:
    """Contours of a binary mask as normalized polygons, which is the label format the seg head reads.

    Small specks are dropped: an anti-aliased ribbon end produces a few stray pixels that become a
    three-point polygon teaching the model nothing.
    """
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        approx = cv2.approxPolyDP(c, epsilon=1.5, closed=True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        flat: list[float] = []
        for x, y in approx:
            flat += [min(max(float(x) / max(width, 1), 0.0), 1.0),
                     min(max(float(y) / max(height, 1), 0.0), 1.0)]
        out.append(flat)
    return out


async def _select(spec: LaneBuildSpec) -> dict[str, dict]:
    """Per frame: its image, its lane rows, and its drivable mask uri."""
    from sqlalchemy import select

    from db.models import DrivableMask, Frame, Lane
    from db.models import Session as DbSession

    frames: dict[str, dict] = {}
    async with get_sessionmaker()() as db:
        if "lane" in spec.sources:
            stmt = (select(Lane, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
                    .join(Frame, Lane.frame_id == Frame.frame_id)
                    .join(DbSession, Frame.session_id == DbSession.session_id))
            if spec.sources_states:
                stmt = stmt.where(Lane.source.in_(spec.sources_states))
            if spec.lane_types:
                stmt = stmt.where(Lane.lane_type.in_(spec.lane_types))
            if spec.cities:
                stmt = stmt.where(DbSession.city.in_(spec.cities))
            for lane, img_uri, w, h, session_id in (await db.execute(stmt)).all():
                f = frames.setdefault(str(lane.frame_id), {
                    "img_uri": img_uri, "w": w, "h": h, "session_id": str(session_id),
                    "lanes": [], "drivable_uri": None})
                f["lanes"].append({"points": lane.control_points or [], "type": lane.lane_type})

        if "drivable" in spec.sources:
            stmt2 = (select(DrivableMask, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
                     .join(Frame, DrivableMask.frame_id == Frame.frame_id)
                     .join(DbSession, Frame.session_id == DbSession.session_id))
            if spec.cities:
                stmt2 = stmt2.where(DbSession.city.in_(spec.cities))
            for dm, img_uri, w, h, session_id in (await db.execute(stmt2)).all():
                f = frames.setdefault(str(dm.frame_id), {
                    "img_uri": img_uri, "w": w, "h": h, "session_id": str(session_id),
                    "lanes": [], "drivable_uri": None})
                f["drivable_uri"] = dm.mask_uri
    return frames


async def build_lane_dataset(spec: LaneBuildSpec) -> dict:
    settings = get_settings()
    store = get_object_store()

    unknown = [s for s in spec.sources if s not in ("lane", "drivable")]
    if unknown:
        raise ValueError(f"unknown lane source(s) {unknown}; use 'lane' and/or 'drivable'")

    out = settings.scratch_path() / "training" / spec.name
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    frames = await _select(spec)
    if not frames:
        raise ValueError(
            "no lane lines or drivable masks match; a lane model needs annotated lanes to learn from")

    # The classes present, in a fixed order so the index in every label line is stable.
    names: list[str] = []
    if "lane" in spec.sources:
        present = {ln["type"] for f in frames.values() for ln in f["lanes"]}
        names += [t for t in LANE_TYPES if t in present]
    if "drivable" in spec.sources:
        names += [f"surface_{c}" for c in DRIVABLE_CLASSES]
    if not names:
        raise ValueError("no lane types are present in the selection")
    idx_of = {n: i for i, n in enumerate(names)}

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

    counts = {"train": 0, "val": 0}
    n_instances = 0
    unreadable = 0
    for fid in fids:
        f = frames[fid]
        try:
            buf = np.frombuffer(store.get_bytes(f["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            unreadable += 1
            continue
        h, w = img.shape[:2]

        lines: list[str] = []
        for lane in f["lanes"]:
            if lane["type"] not in idx_of or len(lane["points"] or []) < spec.min_points:
                continue
            mask = rasterize_lane(lane["points"], w, h, spec.line_width_px)
            if mask is None:
                continue
            for poly in mask_to_polygons(mask, w, h):
                lines.append(f"{idx_of[lane['type']]} " + " ".join(f"{v:.6f}" for v in poly))
                n_instances += 1

        if f["drivable_uri"] and "drivable" in spec.sources:
            lines += _drivable_lines(store, f["drivable_uri"], w, h, idx_of)
            n_instances += len(lines)

        if not lines:
            continue
        split = "val" if f["session_id"] in val_sessions else "train"
        cv2.imwrite(str(out / f"images/{split}/{fid}.jpg"), img)
        (out / f"labels/{split}/{fid}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1

    if counts["train"] == 0:
        raise ValueError("no frame produced a usable lane label; check line_width_px and the lane geometry")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))

    result = {
        "name": spec.name, "dir": str(out), "data_yaml": str(data_yaml),
        "classes": len(names), "class_names": names,
        "n_train_images": counts["train"], "n_val_images": counts["val"],
        "n_instances": n_instances, "unreadable_frames": unreadable,
        "sources": list(spec.sources), "line_width_px": spec.line_width_px,
        "split": "session_grouped" if spec.group_split_by_session else "per_frame",
        "task": "lane",
    }
    log.info("laneset.built", classes=len(names), train=counts["train"], val=counts["val"],
             instances=n_instances)
    return result


def _drivable_lines(store, uri: str, w: int, h: int, idx_of: dict[str, int]) -> list[str]:
    """Turn a stored ternary surface mask into polygon label lines, one per class region."""
    try:
        buf = np.frombuffer(store.get_bytes(uri), dtype=np.uint8)
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001
        return []
    if mask is None:
        return []
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    lines: list[str] = []
    # The stored mask is ternary by value: 0 non-drivable, 1 drivable, 2 fallback, matching the writer in
    # services/perception. Compared exactly rather than thresholded, because a threshold would merge the
    # fallback class into whichever neighbour it sits closer to.
    for value, cls in ((1, "surface_drivable"), (0, "surface_non_drivable"), (2, "surface_fallback")):
        if cls not in idx_of:
            continue
        binary = (mask == value).astype(np.uint8) * 255
        if not binary.any():
            continue
        for poly in mask_to_polygons(binary, w, h, min_area_px=200):
            lines.append(f"{idx_of[cls]} " + " ".join(f"{v:.6f}" for v in poly))
    return lines


_SPEC_FIELDS = {f.name for f in dataclasses.fields(LaneBuildSpec)}


class LaneTask:
    task_type = "lane"

    def default_base_weights(self) -> str:
        # A lane model is a segmentation model over ribbon masks, so it starts from a segmentation
        # checkpoint rather than a detection one: the head shape has to match what is being learned.
        return get_settings().training.segmentation_weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _SPEC_FIELDS}
        spec["name"] = cfg["name"]
        return await build_lane_dataset(LaneBuildSpec(**spec))

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
                          "metrics": {"map50_mask": round(float(m.get("metrics/mAP50(M)", 0.0)), 4)}})
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

        res = YOLO(weights).val(data=data_yaml, imgsz=imgsz, verbose=False)
        seg = getattr(res, "seg", None)
        out = {
            "map50_mask": float(getattr(seg, "map50", 0.0)) if seg else None,
            "map_mask": float(getattr(seg, "map", 0.0)) if seg else None,
        }
        # Gate on the mask number: a lane model whose boxes improve while its ribbons degrade has regressed
        # at the only thing it was trained to do.
        out["map50"] = out["map50_mask"]
        out["map"] = out["map_mask"]
        return out

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        from services.training import eval as eval_mod

        crit = {k: v for k, v in (criteria or {}).items() if k in ("min_map_delta", "max_class_drop")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(LaneTask())
