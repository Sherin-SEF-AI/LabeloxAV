"""Build a YOLO-format training dataset from the labeled corpus for the close-the-loop fine-tune.

Selection encodes the honest lessons about pseudo-labels (see the model-substitution + progress
notes): prefer human-reviewed objects as gold, take cross-path-agreement detections as the
trustworthy pseudo-labels, drop fallback classes, apply a confidence floor, and cap per class to
fight head dominance (balanced sampling). Human-reviewed frames are routed into val so evaluation
is against the cleanest labels available. An external IDD dataset (YOLO format) can be merged in as
the cold-start anchor.
"""

from __future__ import annotations

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

log = get_logger("trainset")


@dataclass
class BuildSpec:
    name: str = "loop-v1"
    conf_floor: float = 0.2
    max_per_class: int = 400        # balanced sampling cap (counters bus_shelter-style over-firing)
    val_frac: float = 0.2
    agreement_only: bool = False    # True = only cross-path-agreement objects (cleanest pseudo-labels)
    # Restrict the trainset to specific object states. Empty, the default, means everything not rejected.
    #
    # Passing the reviewed states (accepted/auto_accept/human/vlm_review) keeps raw `review` labels out of
    # training, which was recorded here as the pollution that collapsed loop-op-v5.
    #
    # There is now measured evidence pointing the other way, and both belong here because they disagree.
    # A controlled comparison on the operational corpus, same frames and session split and validation key
    # and backbone and schedule with only the label states changed, took micro recall from 0.411 to 0.770
    # and rider from 0.000 to 0.468. The reason is that gate-approved objects are 7.6% of what the detector
    # found on those frames, so restricting to them teaches the model that the other 92.4% of the traffic is
    # background: 14,819 riders labelled background against 134 labelled rider.
    #
    # loop-op-v5 could not be re-examined; its job rows are gone. So this is not a claim that the older note
    # was wrong, only that restricting states is not the safe default it reads as, and that anyone reaching
    # for it should measure recall against a fixed key rather than assume cleaner labels are better ones.
    states: list[str] = field(default_factory=list)
    seed: int = 7
    route_prefix: str | None = None  # scope to a capture batch, e.g. "202606"
    cities: list[str] = field(default_factory=list)        # per-domain scoping (e.g. ["BLR"])
    include_classes: list[str] = field(default_factory=list)  # specialized detector: ONLY these classes
    idd_dir: str | None = None       # external YOLO dataset to merge as the IDD anchor
    drop_classes: list[str] = field(default_factory=list)
    # Split validation at the session boundary, not per frame. Consecutive dashcam frames are
    # near-duplicates, so a per-frame split leaks the training distribution into validation and inflates
    # the reported mAP. Set False only to reproduce a historical build.
    group_split_by_session: bool = True
    # Refuse to build if the trainset would contain objects that are also in this sealed gold set. Training
    # on the yardstick makes every gold metric meaningless, and nothing previously prevented it.
    exclude_gold_id: str | None = None


async def _select(spec: BuildSpec):
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession

    onto = get_ontology()
    fallback_ids = set(onto.fallback_ids())
    drop_ids = {onto.by_name(n).id for n in spec.drop_classes if onto.has_name(n)}
    # Specialized detector: restrict to an explicit class allow-list (everything else is dropped).
    include_ids = {onto.by_name(n).id for n in spec.include_classes if onto.has_name(n)}

    maker = get_sessionmaker()
    async with maker() as db:
        stmt = (
            select(Object, Frame.frame_id, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(Object.state != "rejected", Object.conf >= spec.conf_floor)
        )
        if spec.states:
            stmt = stmt.where(Object.state.in_(spec.states))
        if spec.route_prefix:
            stmt = stmt.where(DbSession.route.like(f"{spec.route_prefix}%"))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        rows = (await db.execute(stmt)).all()

    cand = []
    for obj, frame_id, img_uri, w, h, session_id in rows:
        if obj.class_id in fallback_ids or obj.class_id in drop_ids:
            continue
        if include_ids and obj.class_id not in include_ids:
            continue
        agree = bool((obj.provenance or {}).get("agreement"))
        if spec.agreement_only and not agree:
            continue
        is_gold = obj.source == "human" and obj.state == "accepted"
        cand.append({
            "frame_id": str(frame_id), "session_id": str(session_id), "img_uri": img_uri, "w": w, "h": h,
            "class_id": obj.class_id, "bbox": list(obj.bbox), "agree": agree, "gold": is_gold,
            "object_id": str(obj.object_id),
        })
    return cand


def _split_val_frames(by_frame: dict[str, list[dict]], gold_frames: set[str], n_val: int,
                      rng: random.Random, *, group_by_session: bool) -> set[str]:
    """Choose the validation frames, grouping whole sessions together by default.

    A per-frame random split puts consecutive frames of the same drive on both sides. Dashcam frames a
    fraction of a second apart are near-duplicates, so the model effectively validates on images it trained
    on and the reported mAP is optimistic by an unknown margin. Splitting at the session boundary makes
    validation measure generalisation to an unseen drive, which is the number anyone actually wants.

    Sessions are taken whole, smallest-first, until the frame budget is met, so the realised validation size
    lands close to the requested fraction instead of overshooting on one huge session.
    """
    if n_val <= 0:
        return set()
    if not group_by_session:
        frames = list(by_frame)
        rng.shuffle(frames)
        non_gold = [f for f in frames if f not in gold_frames]
        val_set = set(list(gold_frames)[:n_val])
        for f in non_gold:
            if len(val_set) >= n_val:
                break
            val_set.add(f)
        return val_set

    sessions: dict[str, list[str]] = {}
    for fid, objs in by_frame.items():
        sessions.setdefault(objs[0].get("session_id") or "unknown", []).append(fid)

    order = sorted(sessions)
    rng.shuffle(order)
    # Prefer sessions that carry gold frames: gold is the honest yardstick and belongs on the val side.
    order.sort(key=lambda sid: (not any(f in gold_frames for f in sessions[sid]), len(sessions[sid])))

    val_set: set[str] = set()
    for sid in order:
        if len(val_set) >= n_val:
            break
        val_set.update(sessions[sid])
    # With a single session there is nothing to hold out at session granularity; fall back to a frame split
    # rather than emitting an empty validation set (which would silently disable validation entirely).
    if len(sessions) < 2:
        return _split_val_frames(by_frame, gold_frames, n_val, rng, group_by_session=False)
    return val_set


async def _exclude_gold_objects(cand: list[dict], gold_id: str | None) -> tuple[list[dict], int]:
    """Drop any candidate that is a member of the named sealed gold set.

    Training and gold both draw from the same Object table with overlapping predicates (human-reviewed,
    accepted), so the same object could sit in the trainset and in the yardstick at once. A model trained on
    its own gold set scores against memorised examples and every gold metric it produces is meaningless.
    Nothing previously enforced disjointness; naming a gold set here does.
    """
    if not gold_id:
        return cand, 0
    from db.models import GoldSet

    maker = get_sessionmaker()
    async with maker() as db:
        gold = await db.get(GoldSet, gold_id)
    if gold is None:
        raise ValueError(f"exclude_gold_id '{gold_id}' does not exist; refusing to build a trainset that "
                         "cannot be proven disjoint from the yardstick")
    gold_object_ids = {str(o) for o in (gold.object_ids or [])}
    if not gold_object_ids:
        return cand, 0
    kept = [c for c in cand if c.get("object_id") not in gold_object_ids]
    dropped = len(cand) - len(kept)
    if dropped:
        log.info("trainset.gold_excluded", gold_id=gold_id, dropped=dropped)
    return kept, dropped


def _cap_per_class(cand: list[dict], max_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {}
    for c in cand:
        by_class.setdefault(c["class_id"], []).append(c)
    kept = []
    for cid, items in by_class.items():
        # always keep gold; cap the rest
        gold = [i for i in items if i["gold"]]
        rest = [i for i in items if not i["gold"]]
        rng.shuffle(rest)
        room = max(0, max_per_class - len(gold))
        kept.extend(gold + rest[:room])
    return kept


async def build_training_dataset(spec: BuildSpec) -> dict:
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

    # Class vocabulary = union of corpus classes and (if merging) the IDD anchor's classes, so IDD
    # ground-truth classes the corpus barely has (bus, cycle, ...) are kept rather than dropped.
    present_ids = {c["class_id"] for c in cand}
    if spec.idd_dir:
        idd_meta = yaml.safe_load((Path(spec.idd_dir) / "data.yaml").read_text())
        for nm in (idd_meta.get("names") or {}).values():
            if onto.has_name(nm):
                present_ids.add(onto.by_name(nm).id)
    present = sorted(present_ids)
    idx_of = {cid: i for i, cid in enumerate(present)}
    names = {i: onto.by_id(cid).name for cid, i in idx_of.items()}

    # group by frame; mark gold frames
    by_frame: dict[str, list[dict]] = {}
    gold_frames: set[str] = set()
    for c in cand:
        by_frame.setdefault(c["frame_id"], []).append(c)
        if c["gold"]:
            gold_frames.add(c["frame_id"])

    frames = list(by_frame)
    rng = random.Random(spec.seed)
    n_val = max(1, int(len(frames) * spec.val_frac)) if len(frames) > 4 else 0
    val_set = _split_val_frames(by_frame, gold_frames, n_val, rng, group_by_session=spec.group_split_by_session)

    n_train_obj = n_val_obj = 0
    for fid, objs in by_frame.items():
        split = "val" if fid in val_set else "train"
        first = objs[0]
        try:
            buf = np.frombuffer(store.get_bytes(first["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue  # synthetic/test frames with no real blob are skipped
        cv2.imwrite(str(out / f"images/{split}/{fid}.jpg"), img)
        lines = []
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            w, h = max(1, o["w"]), max(1, o["h"])
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{idx_of[o['class_id']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / f"labels/{split}/{fid}.txt").write_text("\n".join(lines) + "\n")
        if split == "train":
            n_train_obj += len(objs)
        else:
            n_val_obj += len(objs)

    if spec.idd_dir:
        _merge_idd(Path(spec.idd_dir), out, names, idx_of, onto)

    n_train_imgs = len(list((out / "images/train").glob("*.jpg")))
    n_val_imgs = len(list((out / "images/val").glob("*.jpg")))
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": names,
    }, sort_keys=False))

    result = {
        "name": spec.name, "dir": str(out), "data_yaml": str(data_yaml),
        "classes": len(names), "n_train_images": n_train_imgs, "n_val_images": n_val_imgs,
        "n_train_objects": n_train_obj, "n_val_objects": n_val_obj,
        "gold_frames": len(gold_frames), "ontology_version": onto.version,
        # Provenance of the split itself, so a reported metric can be traced to how it was validated.
        "split": "session_grouped" if spec.group_split_by_session else "per_frame",
        "val_sessions": len({by_frame[f][0].get("session_id") for f in val_set}) if val_set else 0,
        "excluded_gold_id": spec.exclude_gold_id,
        "excluded_gold_objects": excluded_gold,
    }
    log.info("trainset.built", **{k: result[k] for k in ("classes", "n_train_images", "n_val_images")})
    return result


def _merge_idd(idd_dir: Path, out: Path, names: dict, idx_of: dict, onto) -> None:
    """Merge an external IDD YOLO dataset (the cold-start anchor). Expects idd_dir/data.yaml with
    names; remaps its class indices into our ontology by name where they match."""
    idd_yaml = idd_dir / "data.yaml"
    if not idd_yaml.exists():
        log.warning("idd.missing_yaml", dir=str(idd_dir))
        return
    idd = yaml.safe_load(idd_yaml.read_text())
    idd_names = idd.get("names", {})
    if isinstance(idd_names, list):
        idd_names = dict(enumerate(idd_names))
    # build remap idd_idx -> our_idx by matching class names present in our ontology
    name_to_our: dict[str, int] = {v: k for k, v in names.items()}
    remap: dict[int, int] = {}
    for i, nm in idd_names.items():
        if nm in name_to_our:
            remap[int(i)] = name_to_our[nm]
    copied = 0
    for split in ("train", "val"):
        img_dir = idd_dir / "images" / split
        lab_dir = idd_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.*"):
            lab = lab_dir / (img.stem + ".txt")
            if not lab.exists():
                continue
            kept = []
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                ci = int(parts[0])
                if ci in remap:
                    kept.append(f"{remap[ci]} {' '.join(parts[1:])}")
            if not kept:
                continue
            dst = out / f"images/{split}/idd_{img.name}"
            if not dst.exists():
                dst.symlink_to(img.resolve())  # symlink, not copy: IDD is ~22 GB
            (out / f"labels/{split}/idd_{img.stem}.txt").write_text("\n".join(kept) + "\n")
            copied += 1
    log.info("idd.merged", images=copied, mapped_classes=len(remap))
