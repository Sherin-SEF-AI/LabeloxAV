"""Build a YOLO-segmentation dataset from the labeled corpus.

The detection builder writes `cls cx cy w h` box lines and drops every mask, so the polygons a reviewer
traced with SAM assistance could be exported but never trained on. This writes the YOLO-seg label format
(`cls x1 y1 x2 y2 ... xn yn`, normalized polygon vertices), which is what closes the loop for segmentation.

It reuses the detection builder's selection discipline rather than re-deriving it: the same confidence floor,
the same state filter, the same session-grouped split that stops near-duplicate dashcam frames leaking across
it, and the same gold-exclusion guard. The only differences are that an object without a mask is not a
candidate, and the label line carries the polygon.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.dataset_builder import _exclude_gold_objects, _split_val_frames

log = get_logger("segset")


@dataclass
class SegBuildSpec:
    name: str = "seg-v1"
    conf_floor: float = 0.2
    max_per_class: int = 400
    val_frac: float = 0.2
    states: list[str] = field(default_factory=list)
    seed: int = 7
    cities: list[str] = field(default_factory=list)
    include_classes: list[str] = field(default_factory=list)
    drop_classes: list[str] = field(default_factory=list)
    group_split_by_session: bool = True
    exclude_gold_id: str | None = None
    # A polygon with too few vertices is a degenerate mask (often a failed SAM call collapsed to a sliver).
    # Training on those teaches the model to emit slivers, so they are dropped rather than normalized.
    min_polygon_points: int = 3


def _load_polygons(store, mask_uri: str | None) -> list[list[float]]:
    """Masks are stored as a JSON polygon list in object storage, the same shape the exporters read."""
    if not mask_uri:
        return []
    import json

    try:
        return json.loads(store.get_bytes(mask_uri)).get("polygons", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("segset.mask_load_failed", uri=mask_uri, error=str(exc))
        return []


async def _select(spec: SegBuildSpec) -> list[dict]:
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession

    onto = get_ontology()
    fallback_ids = set(onto.fallback_ids())
    drop_ids = {onto.by_name(n).id for n in spec.drop_classes if onto.has_name(n)}
    include_ids = {onto.by_name(n).id for n in spec.include_classes if onto.has_name(n)}

    async with get_sessionmaker()() as db:
        stmt = (
            select(Object, Frame.frame_id, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            # A mask is the label here, so an object without one is not a candidate at all.
            .where(Object.state != "rejected", Object.conf >= spec.conf_floor,
                   Object.mask_uri.isnot(None))
        )
        if spec.states:
            stmt = stmt.where(Object.state.in_(spec.states))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        rows = (await db.execute(stmt)).all()

    store = get_object_store()
    cand: list[dict] = []
    for obj, frame_id, img_uri, w, h, session_id in rows:
        if obj.class_id in fallback_ids or obj.class_id in drop_ids:
            continue
        if include_ids and obj.class_id not in include_ids:
            continue
        polys = _load_polygons(store, obj.mask_uri)
        polys = [p for p in polys if len(p) >= spec.min_polygon_points * 2]
        if not polys:
            continue
        cand.append({
            "frame_id": str(frame_id), "session_id": str(session_id), "object_id": str(obj.object_id),
            "img_uri": img_uri, "w": w, "h": h, "class_id": obj.class_id,
            "polygons": polys, "gold": obj.source == "human" and obj.state == "accepted",
        })
    return cand


def _cap_per_class(cand: list[dict], max_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {}
    for c in cand:
        by_class.setdefault(c["class_id"], []).append(c)
    kept: list[dict] = []
    for items in by_class.values():
        gold = [i for i in items if i["gold"]]
        rest = [i for i in items if not i["gold"]]
        rng.shuffle(rest)
        kept.extend(gold + rest[:max(0, max_per_class - len(gold))])
    return kept


def polygon_label_line(class_index: int, polygon: list[float], width: int, height: int) -> str | None:
    """One YOLO-seg label line: class index then normalized x y pairs, clamped into the image.

    Ultralytics reads one polygon per line, so a multi-part mask becomes several lines of the same class,
    which is also how it handles an object split by an occluder.
    """
    pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return None
    xs = np.clip(pts[:, 0] / max(width, 1), 0.0, 1.0)
    ys = np.clip(pts[:, 1] / max(height, 1), 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys, strict=True))
    return f"{class_index} {coords}"


async def build_segmentation_dataset(spec: SegBuildSpec) -> dict:
    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    out = settings.scratch_path() / "training" / spec.name
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cand = await _select(spec)
    cand, excluded_gold = await _exclude_gold_objects(cand, spec.exclude_gold_id)
    cand = _cap_per_class(cand, spec.max_per_class, spec.seed)
    if not cand:
        raise ValueError("no masked objects matched the spec; segmentation training needs mask labels")

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

    n_train_obj = n_val_obj = 0
    n_train_img = n_val_img = 0
    last_uri, last_img = None, None
    for fid, objs in by_frame.items():
        split = "val" if fid in val_set else "train"
        first = objs[0]
        if first["img_uri"] != last_uri:
            try:
                buf = np.frombuffer(store.get_bytes(first["img_uri"]), dtype=np.uint8)
                last_uri, last_img = first["img_uri"], cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001 - synthetic/test frames with no blob are skipped
                last_uri, last_img = first["img_uri"], None
        if last_img is None:
            continue

        lines: list[str] = []
        for o in objs:
            for poly in o["polygons"]:
                line = polygon_label_line(idx_of[o["class_id"]], poly, o["w"], o["h"])
                if line:
                    lines.append(line)
        if not lines:
            continue
        cv2.imwrite(str(out / f"images/{split}/{fid}.jpg"), last_img)
        (out / f"labels/{split}/{fid}.txt").write_text("\n".join(lines) + "\n")
        if split == "train":
            n_train_img += 1
            n_train_obj += len(objs)
        else:
            n_val_img += 1
            n_val_obj += len(objs)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": names,
    }, sort_keys=False))

    result = {
        "name": spec.name, "dir": str(out), "data_yaml": str(data_yaml),
        "classes": len(names), "n_train_images": n_train_img, "n_val_images": n_val_img,
        "n_train_objects": n_train_obj, "n_val_objects": n_val_obj,
        "gold_frames": len(gold_frames), "ontology_version": onto.version,
        "split": "session_grouped" if spec.group_split_by_session else "per_frame",
        "excluded_gold_id": spec.exclude_gold_id, "excluded_gold_objects": excluded_gold,
        "task": "segmentation",
    }
    log.info("segset.built", **{k: result[k] for k in ("classes", "n_train_images", "n_val_images")})
    return result
