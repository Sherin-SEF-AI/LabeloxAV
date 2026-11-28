"""How accurate is auto-labelling, on Indian frames where every object is labelled?

    .venv/bin/python scripts/idd_pipeline_eval.py --n 300 [--weights PATH] [--agent] --tag champion

The corpus cannot answer this: people label some objects in a frame, not all, so a machine box with no
human box under it might be a mistake or might be an object nobody drew. Precision is unmeasurable there.
IDD Segmentation's validation split (2,036 frames, Hyderabad and Bangalore) draws an outline around every
instance of its thing classes, so every machine box can be graded.

The run is the product, not one model: frames go in through ingest (privacy blur included), the full
auto-label pipeline runs (both detectors, fusion, calibration, gate, reasoner), optionally the label agent,
and the objects as stored are scored. Each run makes its own session, `IDD-EVAL-<tag>`.

Scoring is at IDD's level of detail, which is coarser than the ontology: a hatchback and an SUV are both
`car`. Rules:
- Ground truth boxes are the outline's extent, rescaled to the stored frame. Boxes under 16 px tall or 8 px
  wide are ignored both ways (neither a miss nor a false positive), as are IDD's catch-alls (`vehicle
  fallback`, `caravan`, `polegroup`, barrier/structure fallback): a machine box on one of those is not
  counted against the machine.
- A machine box is right when it overlaps an unmatched ground-truth box of the same category at IoU >= 0.5
  (greedy by confidence). Ontology classes IDD has no category for (a tree, a shop, a manhole) are left out
  of precision rather than counted wrong, and reported separately.
- Precision is reported per gate outcome. `auto_accept` precision is the number a buyer asks for: of the
  labels nobody will look at, how many are right.

Caveat to state with every number: the serving champion was trained on IDD *Detection*. That is a separate
release from IDD Segmentation, but both come from the same fleet and cities, so frames may overlap. Numbers
for the champion may be optimistic; numbers for a model not trained on IDD are not affected.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IDD_ROOT = Path("/home/jo/lake/idd/segmentation")
BASE_TS = 1_600_000_000_000_000_000
STEP_NS = 333_333_333
MIN_H, MIN_W = 16, 8

GT_CATEGORY = {"car": "car", "bus": "bus", "truck": "truck", "motorcycle": "motorcycle", "bicycle": "bicycle",
               "autorickshaw": "autorickshaw", "person": "person", "rider": "rider", "animal": "animal",
               "traffic sign": "traffic_sign", "traffic light": "traffic_light", "billboard": "billboard",
               "pole": "pole"}
GT_IGNORE = {"vehicle fallback", "caravan", "trailer", "train", "polegroup", "obs-str-bar-fallback"}
VEHICLES = {"car", "bus", "truck"}

BY_NAME = {
    "bus": "bus", "school_bus": "bus",
    "cycle": "bicycle", "cargo_bike": "bicycle", "cyclist": "rider", "rider": "rider",
    "delivery_rider_bike": "motorcycle", "scooter_with_rider": "motorcycle", "two_wheeler": "motorcycle",
    "back_side_of_autorickshaw": "autorickshaw", "back_side_of_mini_lorry": "truck", "mini_van": "car",
    "traffic_sign": "traffic_sign", "chevron_sign": "traffic_sign", "non_parking_sign": "traffic_sign",
    "traffic_signal": "traffic_light", "pedestrian_signal": "traffic_light",
    "hoarding": "billboard", "flex_banner": "billboard", "billboard": "billboard",
    "advertisement_board": "billboard", "name_board": "billboard",
    "pole": "pole", "street_light": "pole", "electric_pole": "pole", "electric_post": "pole",
    "light_pole": "pole", "traffic_pole": "pole", "signal_pole": "pole", "cctv_pole": "pole",
    # vans and the vehicle fallback are a car, a truck or a minibus depending on who is asked: lenient
    "vehicle_fallback": "vehicle_any", "milk_van": "vehicle_any", "police_van": "vehicle_any",
    "ambulance": "vehicle_any", "school_van": "vehicle_any", "tempo_traveller": "vehicle_any",
    "cycle_rickshaw": "vehicle_any",
}
BY_FAMILY = {"four_wheeler": "car", "heavy": "truck", "two_wheeler": "motorcycle",
             "three_wheeler": "autorickshaw", "animal": "animal", "vru": "person"}


def category(name: str, family: str) -> str | None:
    if name in BY_NAME:
        return BY_NAME[name]
    return BY_FAMILY.get(family)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pick(n: int, seed: int) -> list[Path]:
    files = sorted(glob.glob(str(IDD_ROOT / "gtFine" / "val" / "*" / "*_gtFine_polygons.json")))
    random.Random(seed).shuffle(files)
    return [Path(f) for f in files[:n]]


def image_for(poly: Path) -> Path:
    stem = poly.name.replace("_gtFine_polygons.json", "")
    seq = poly.parent.name
    for ext in ("png", "jpg"):
        p = IDD_ROOT / "leftImg8bit" / "val" / seq / f"{stem}_leftImg8bit.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(poly)


def ground_truth(poly: Path, scale: float) -> tuple[list[dict], list[list[float]]]:
    j = json.loads(poly.read_text())
    keep, ignore = [], []
    for o in j["objects"]:
        if o.get("deleted"):
            continue
        pts = o.get("polygon") or []
        if len(pts) < 3:
            continue
        xs = [p[0] * scale for p in pts]
        ys = [p[1] * scale for p in pts]
        box = [min(xs), min(ys), max(xs), max(ys)]
        lab = o["label"]
        if lab in GT_IGNORE:
            ignore.append(box)
        elif lab in GT_CATEGORY:
            small = (box[3] - box[1]) < MIN_H or (box[2] - box[0]) < MIN_W
            keep.append({"cat": GT_CATEGORY[lab], "bbox": box, "small": small})
    return keep, ignore


async def ingest_sample(polys: list[Path], tag: str) -> str:
    import cv2

    from services.ingest.run import ingest
    from services.ingest.types import RawFrame

    def frames():
        for i, p in enumerate(polys):
            img = cv2.imread(str(image_for(p)))
            if img is not None:
                yield RawFrame(ts_ns=BASE_TS + i * STEP_NS, cam_id="cam_front", image_bgr=img)

    res = await ingest(frame_iter=frames(), vehicle=f"IDD-EVAL-{tag}", city="IDD", route="idd-seg-val",
                       raw_uri=None, mcap_uri=None, source_streams=["idd_segmentation_val"])
    return str(res["session_id"])


async def snapshot(sid: str) -> dict[int, dict]:
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        frames = (await db.execute(select(Frame.frame_id, Frame.ts_ns, Frame.width)
                                   .where(Frame.session_id == uuid.UUID(sid)))).all()
        objs = (await db.execute(select(Object.frame_id, Object.class_id, Object.bbox, Object.conf, Object.state,
                                        Object.source)
                                 .join(Frame, Frame.frame_id == Object.frame_id)
                                 .where(Frame.session_id == uuid.UUID(sid)))).all()
    by_frame = defaultdict(list)
    for o in objs:
        by_frame[o.frame_id].append(o)
    out = {}
    for f in frames:
        i = round((int(f.ts_ns) - BASE_TS) / STEP_NS)
        out[i] = {"width": f.width, "objects": by_frame.get(f.frame_id, [])}
    return out


def score(polys: list[Path], snap: dict[int, dict], onto) -> dict:
    name = {c.id: c.name for c in onto.classes}
    fam = {c.id: c.l1 for c in onto.classes}
    gt_n = gt_found = gt_class = 0
    per_cat = defaultdict(lambda: Counter())
    bucket = defaultdict(Counter)
    outside = Counter()
    confusions = Counter()
    frames_scored = 0
    for i, poly in enumerate(polys):
        if i not in snap:
            continue
        j = json.loads(poly.read_text())
        scale = (snap[i]["width"] or j["imgWidth"]) / j["imgWidth"]
        gts, ignores = ground_truth(poly, scale)
        frames_scored += 1
        preds = []
        for o in snap[i]["objects"]:
            if not o.bbox or len(o.bbox) != 4:
                continue
            cat = category(name.get(int(o.class_id), ""), fam.get(int(o.class_id), ""))
            b = [float(v) for v in o.bbox]
            state = o.state if o.state in ("auto_accept", "review", "annotate") else "other"
            if cat is None:
                outside[name.get(int(o.class_id), "?")] += 1
                continue
            small = (b[3] - b[1]) < MIN_H or (b[2] - b[0]) < MIN_W
            preds.append({"cat": cat, "bbox": b, "conf": float(o.conf or 0), "state": state, "small": small,
                          "name": name.get(int(o.class_id), "?")})
        # class-aware greedy matching
        matched_gt: set[int] = set()
        preds.sort(key=lambda p: -p["conf"])
        for p in preds:
            best, bi = 0.0, -1
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                ok = g["cat"] == p["cat"] or (p["cat"] == "vehicle_any" and g["cat"] in VEHICLES)
                if ok:
                    v = iou(p["bbox"], g["bbox"])
                    if v > best:
                        best, bi = v, gi
            if best >= 0.5:
                matched_gt.add(bi)
                p["tp"] = True
                if gts[bi]["small"]:
                    p["ignored"] = True                     # a hit on an ignored box is neither way
            else:
                p["tp"] = False
                overlaps_any = max((iou(p["bbox"], g["bbox"]) for g in gts), default=0.0)
                on_ignore = any(iou(p["bbox"], r) >= 0.5 or
                                _ioa(p["bbox"], r) >= 0.7 for r in ignores)
                if p["small"] or on_ignore:
                    p["ignored"] = True
                elif overlaps_any >= 0.5:
                    g = max(gts, key=lambda g: iou(p["bbox"], g["bbox"]))
                    if not g["small"]:
                        confusions[f'{p["name"]} on {g["cat"]}'] += 1
        for p in preds:
            if p.get("ignored"):
                continue
            bucket[p["state"]]["n"] += 1
            bucket[p["state"]]["right"] += p["tp"]
            per_cat[p["cat"]]["pred"] += 1
            per_cat[p["cat"]]["pred_right"] += p["tp"]
        # recall, class-aware and class-agnostic
        for gi, g in enumerate(gts):
            if g["small"]:
                continue
            gt_n += 1
            per_cat[g["cat"]]["gt"] += 1
            if gi in matched_gt:
                gt_class += 1
                per_cat[g["cat"]]["gt_found_right_class"] += 1
            all_boxes = [p["bbox"] for p in preds]
            if any(iou(g["bbox"], b) >= 0.5 for b in all_boxes):
                gt_found += 1

    def r(k, n):
        return round(k / n, 4) if n else None

    allp = sum(b["n"] for b in bucket.values())
    allr = sum(b["right"] for b in bucket.values())
    return {
        "frames": frames_scored, "objects_in_truth": gt_n,
        "recall_right_class": r(gt_class, gt_n), "recall_any_class": r(gt_found, gt_n),
        "precision_all_labels": r(allr, allp), "labels_scored": allp,
        "by_gate_outcome": {k: {"labels": v["n"], "precision": r(v["right"], v["n"]),
                                "share_of_labels": r(v["n"], allp)} for k, v in sorted(bucket.items())},
        "per_category": {k: {"truth": v["gt"], "recall": r(v["gt_found_right_class"], v["gt"]),
                             "labels": v["pred"], "precision": r(v["pred_right"], v["pred"])}
                         for k, v in sorted(per_cat.items(), key=lambda kv: -kv[1]["gt"])},
        "top_confusions": confusions.most_common(15),
        "labels_outside_idd_categories": sum(outside.values()),
        "outside_top": outside.most_common(10),
    }


def _ioa(a, region) -> float:
    ix = max(0.0, min(a[2], region[2]) - max(a[0], region[0]))
    iy = max(0.0, min(a[3], region[3]) - max(a[1], region[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", default=None, help="detector weights; default is the serving champion")
    ap.add_argument("--session", default=None, help="use an already-ingested IDD-EVAL session")
    ap.add_argument("--score-only", action="store_true", help="with --session: score what is stored, no labelling")
    ap.add_argument("--resume", action="store_true", help="with --session: label only frames with no objects yet")
    ap.add_argument("--agent", action="store_true", help="also run the label agent and score after it")
    ap.add_argument("--out", default=".scratch/idd_eval")
    ap.add_argument("--allow-office-hours", action="store_true")
    args = ap.parse_args()

    from services.tools.registry import in_quiet_hours
    if in_quiet_hours() and not args.allow_office_hours and not args.score_only:
        print("auto-labelling waits for office hours to end (10:00-18:00 IST)", file=sys.stderr)
        return 2

    from services.autolabel.ontology import get_ontology
    onto = get_ontology()
    polys = pick(args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sid = args.session
    timings = {}
    if sid is None:
        sid = await ingest_sample(polys, args.tag)
        timings["ingest_s"] = round(time.time() - t0, 1)
        print("ingested", sid, flush=True)
    if not args.score_only:
        from services.autolabel.runner import autolabel_session
        t1 = time.time()
        counts = await autolabel_session(uuid.UUID(sid), None, detector_weights=args.weights,
                                         only_unlabelled=args.resume)
        timings["autolabel_s"] = round(time.time() - t1, 1)
        print("autolabelled", {k: counts.get(k) for k in ("n_frames", "n_objects")}, flush=True)
    report = {"tag": args.tag, "session": sid, "weights": args.weights or "serving champion",
              "n": args.n, "seed": args.seed, "timings": timings}
    report["pipeline"] = score(polys, await snapshot(sid), onto)
    print(json.dumps(report["pipeline"], indent=1)[:3000], flush=True)
    if args.agent:
        from db.session import get_sessionmaker
        from services.intelligence.embed.service import embed_objects
        from services.labelagent.agent import run_session

        await embed_objects(uuid.UUID(sid), only_missing=True)
        async with get_sessionmaker()() as db:
            res = await run_session(db, sid, apply=True, created_by="idd-eval")
        report["agent_run"] = {k: v for k, v in res.items() if k != "samples"}
        report["pipeline_plus_agent"] = score(polys, await snapshot(sid), onto)
        print(json.dumps(report["pipeline_plus_agent"], indent=1)[:2000], flush=True)
    path = out / f"{args.tag}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
