"""Build a detection training set from the corpus that carries what people ruled, without teaching the
detector the autolabeler's own junk.

    .venv/bin/python scripts/build_ruled_trainset.py --out .scratch/training/ruled-v1 [--dry-run]

Why a new builder. The serving champion was trained on IDD alone: 15 classes, and none of the 4,106 human
rulings in this corpus. It fires again on 86% of the boxes people said were not objects, and cannot name a
hoarding, a pole or a barrier at all. Two earlier lessons shape the recipe:

- **Partial annotation** (memory: partial-annotation-was-the-constraint). Training only on reviewed boxes
  teaches the other 92% of traffic as background. So unreviewed machine boxes are used, but only the
  confident ones (both paths agree, or one path is confident). The rest are neither taught as objects nor as
  background: their pixels are painted grey, which is how an ignore region is expressed to a detector that
  has no ignore label.
- **The human ruling wins.** A box a person reclassified carries the person's class; a box a person said is
  not an object stays in the image, unpainted and unlabelled, so the detector is taught it is background.
  That is the only genuine negative evidence in the corpus and the point of this retrain.

Held out, so the comparison afterwards is honest: every frame in a test-split block (the label agent's
30-minute vehicle blocks, so neighbouring frames never straddle the split), every gold-set frame, and whole
vehicles whose frames may duplicate others (ANNO-01 is MCAP re-imports) or are not Indian dashcam footage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAIN_VEHICLES = ("DASHCAM-01", "BLURABBIT-RIG-01", "INDIA-CTC-01", "INDIA-DEL-01", "DEMO-DASHCAM",
                  "dashcam2", "dashcam3", "BLR-DEMO-01", "BLR-DEMO-02", "BLR-TOUR", "TIGOR-DEMO")
HUMAN_STATES = {"accepted", "submitted"}
HUMAN_SOURCES = {"human", "vlm_review"}
# Things the box detector should not learn: surfaces and markings are segmentation's job, and the
# not-an-object sentinels are answers, not classes.
NOT_BOXES = {"lane_marking", "undefined", "background", "road", "drivable", "sky", "pothole", "speed_breaker",
             "crosswalk", "zebra_crossing", "stop_line", "road_marking"}
GREY = 114


def confident(prov: dict) -> bool:
    """An unreviewed machine box worth teaching: the paths agree, or one of them is sure."""
    raw = prov.get("raw_conf") or {}
    a = max([v for k, v in raw.items() if k.startswith("path_a")] or [0.0])
    b = max([v for k, v in raw.items() if not k.startswith("path_a")] or [0.0])
    if (prov.get("reasoning") or {}).get("decision") == "reject":
        return False
    return bool(prov.get("agreement")) or a >= 0.5 or b >= 0.4


async def collect(min_instances: int, stride: int) -> dict:
    from sqlalchemy import select

    from db.models import Frame, GoldSet, Object, Session
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelagent.truth import NOT_OBJECT, block_key, load_truth, split_of

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        truth = {i.object_id: i for i in await load_truth(db)}
        gold_ids = set()
        for (ids,) in (await db.execute(select(GoldSet.object_ids))).all():
            gold_ids.update(ids or [])
        gold_frames = set()
        gl = [uuid.UUID(i) for i in gold_ids]
        for k in range(0, len(gl), 2000):
            gold_frames.update(str(f) for (f,) in (await db.execute(
                select(Object.frame_id).where(Object.object_id.in_(gl[k:k + 2000])))).all())
        frames = (await db.execute(
            select(Frame.frame_id, Frame.img_uri, Frame.ts_ns, Frame.width, Frame.height, Session.vehicle_id)
            .join(Session, Session.session_id == Frame.session_id)
            .where(Session.vehicle_id.in_(TRAIN_VEHICLES)).order_by(Session.vehicle_id, Frame.ts_ns))).all()
        truth_frames = {i.frame_id for i in truth.values()}
        kept, seen = [], Counter()
        split_n = Counter()
        for f in frames:
            fid = str(f.frame_id)
            side = split_of(block_key(str(f.vehicle_id), int(f.ts_ns or 0)))
            split_n[side] += 1
            if side != "calib" or fid in gold_frames or not f.img_uri:
                continue
            seen[f.vehicle_id] += 1
            if fid in truth_frames or seen[f.vehicle_id] % stride == 0:
                kept.append(f)
        objs: dict[str, list] = defaultdict(list)
        ids = [f.frame_id for f in kept]
        for k in range(0, len(ids), 1000):
            for o in (await db.execute(
                    select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox, Object.state,
                           Object.source, Object.provenance)
                    .where(Object.frame_id.in_(ids[k:k + 1000])))).all():
                objs[str(o.frame_id)].append(o)

    name = {c.id: c.name for c in onto.classes}
    fam = {c.id: c.l1 for c in onto.classes}
    plan, why = [], Counter()
    for f in kept:
        pos, ignore = [], []
        for o in objs.get(str(f.frame_id), []):
            if not o.bbox or len(o.bbox) != 4:
                continue
            t = truth.get(str(o.object_id))
            cname = name.get(int(o.class_id), "")
            if o.state == "rejected" or (t is not None and t.truth == NOT_OBJECT):
                why["negative (person said not an object)"] += 1
                continue                                     # stays in the picture as background
            if cname in NOT_BOXES or not cname:
                why["not a box class"] += 1
                ignore.append(o.bbox)
                continue
            human = (t is not None or o.state in HUMAN_STATES or o.source in HUMAN_SOURCES)
            if human:
                why["human ruled or drawn"] += 1
                pos.append((int(t.truth) if t is not None else int(o.class_id), o.bbox))
            elif confident(o.provenance or {}):
                why["confident machine box"] += 1
                pos.append((int(o.class_id), o.bbox))
            else:
                why["unsure machine box -> ignore region"] += 1
                ignore.append(o.bbox)
        plan.append({"frame": f, "pos": pos, "ignore": ignore})

    counts = Counter(name[c] for p in plan for c, _ in p["pos"])
    classes = sorted(n for n, k in counts.items() if k >= min_instances)
    # rare classes fold into their family's fallback rather than vanish (vanishing teaches background)
    fallback = {}
    for cid, n in name.items():
        if n not in classes:
            fallback[cid] = "vehicle_fallback" if fam.get(cid) in {"four_wheeler", "heavy", "two_wheeler",
                                                                  "three_wheeler"} else "object_fallback"
    for fb in ("vehicle_fallback", "object_fallback"):
        if fb not in classes:
            classes.append(fb)
    return {"plan": plan, "classes": classes, "name": name, "fallback": fallback, "counts": counts,
            "why": why, "split_frames": split_n, "gold_frames_excluded": len(gold_frames)}


def write(data: dict, out: Path, val_fraction: float, seed: int) -> None:
    import cv2
    import numpy as np

    from core.storage import get_object_store

    store = get_object_store()
    classes = data["classes"]
    idx = {n: i for i, n in enumerate(classes)}
    name, fallback = data["name"], data["fallback"]
    plan = data["plan"]
    rng = random.Random(seed)
    # val by whole session-ish runs: every 25th frame block of 40 consecutive frames, so val is not the
    # neighbour of a train frame one third of a second away
    val = set()
    for i in range(0, len(plan), 40):
        if rng.random() < val_fraction:
            val.update(range(i, min(i + 40, len(plan))))
    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    def one(i: int) -> str:
        p = plan[i]
        f = p["frame"]
        sub = "val" if i in val else "train"
        stem = str(f.frame_id)
        img_path = out / "images" / sub / f"{stem}.jpg"
        if img_path.exists():
            return "cached"
        try:
            img = cv2.imdecode(np.frombuffer(store.get_bytes(f.img_uri), np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return "unreadable"
        if img is None:
            return "unreadable"
        h, w = img.shape[:2]
        if p["ignore"]:
            m = np.zeros((h, w), bool)
            for x1, y1, x2, y2 in p["ignore"]:
                if (x2 - x1) * (y2 - y1) > 0.2 * w * h:
                    continue                                 # greying a fifth of the frame costs more than it saves
                m[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = True
            for _, (x1, y1, x2, y2) in p["pos"]:
                m[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = False
            img[m] = GREY
        lines = []
        for cid, (x1, y1, x2, y2) in p["pos"]:
            n = name.get(cid, "")
            n = n if n in idx else fallback.get(cid, "object_fallback")
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{idx[n]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} "
                         f"{(y2 - y1) / h:.6f}")
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return sub

    done = Counter()
    with ThreadPoolExecutor(12) as ex:
        for k, r in enumerate(ex.map(one, range(len(plan)))):
            done[r] += 1
            if k % 1000 == 0:
                print(f"  {k}/{len(plan)} {dict(done)}", flush=True)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)))
    print("written:", dict(done))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".scratch/training/ruled-v1")
    ap.add_argument("--min-instances", type=int, default=300)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame (frames with rulings always)")
    ap.add_argument("--val-fraction", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = asyncio.run(collect(args.min_instances, args.stride))
    summary = {"frames": len(data["plan"]), "split_frames_all_vehicles": dict(data["split_frames"]),
               "gold_frames_excluded": data["gold_frames_excluded"], "boxes": dict(data["why"]),
               "classes": data["classes"],
               "class_counts": {n: data["counts"][n] for n in data["classes"] if n in data["counts"]},
               "folded_into_fallback": sum(k for n, k in data["counts"].items() if n not in data["classes"])}
    print(json.dumps(summary, indent=1))
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "build.json").write_text(json.dumps({**summary, "args": vars(args)}, indent=1))
    write(data, out, args.val_fraction, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
