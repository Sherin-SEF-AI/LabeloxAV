"""Compare detectors on what people labelled: the serving champion, Path B, and Path D's two backends.

    .venv/bin/python scripts/detector_benchmark.py --max-frames 400 --save

Three measurements, each chosen because it survives partial annotation (a gold frame labels some objects,
not all, so an unlabelled object a detector finds is not a false positive and precision is not measurable):

- **Gold recall**: of the boxes people drew and accepted (the gold sets), the share a detector finds at
  IoU >= 0.5 with the right class, and class-agnostic. Per class, and for small objects (< 32x32 px).
- **Class on ruled boxes**: of the machine boxes people kept but re-labelled, the share where the detector's
  best-overlapping box carries the person's class. This is the "inaccurate labels" complaint measured
  directly, on exactly the objects people found wrong.
- **Junk re-fire**: of the boxes people said were not objects, the share a detector fires on again. Lower
  is better.

Detectors run one at a time and are unloaded in between (16 GB card). Detections are cached under the
scratch directory so re-scoring does not re-run models. GPU: refuses to start during office hours.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DETECTORS = ["champion", "path_b", "gdino", "owlv2"]
SMALL = 32 * 32


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


async def frame_sets(max_frames: int, seed: int = 7, test_only: bool = False):
    """Gold boxes and ruled boxes, grouped by frame, sampled to the frame budget (half each).

    test_only keeps frames from the label agent's test-split blocks alone: the ones a detector retrained on
    the corpus (scripts/build_ruled_trainset.py) never saw, nor any frame within the same 30 minutes."""
    from sqlalchemy import select

    from db.models import Frame, GoldSet, Object, Session
    from db.session import get_sessionmaker
    from services.labelagent.truth import NOT_OBJECT, block_key, load_truth, split_of

    async with get_sessionmaker()() as db:
        gold_ids = set()
        for (ids,) in (await db.execute(select(GoldSet.object_ids))).all():
            gold_ids.update(ids or [])
        gold: dict[str, list] = defaultdict(list)
        ids = [uuid.UUID(i) for i in gold_ids]
        for k in range(0, len(ids), 2000):
            for _oid, fid, cid, bbox in (await db.execute(
                    select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
                    .where(Object.object_id.in_(ids[k:k + 2000])))).all():
                gold[str(fid)].append({"class_id": int(cid), "bbox": [float(v) for v in bbox]})
        ruled: dict[str, list] = defaultdict(list)
        for it in await load_truth(db):
            if it.machine_made:
                ruled[it.frame_id].append({"truth": it.truth, "pred": it.pred, "bbox": it.bbox,
                                           "not_object": it.truth == NOT_OBJECT})
        if test_only:
            side = {}
            fids = [uuid.UUID(f) for f in set(gold) | set(ruled)]
            for k in range(0, len(fids), 2000):
                for fid, ts, veh in (await db.execute(
                        select(Frame.frame_id, Frame.ts_ns, Session.vehicle_id)
                        .join(Session, Session.session_id == Frame.session_id)
                        .where(Frame.frame_id.in_(fids[k:k + 2000])))).all():
                    side[str(fid)] = split_of(block_key(str(veh), int(ts or 0)))
            gold = defaultdict(list, {f: v for f, v in gold.items() if side.get(f) == "test"})
            ruled = defaultdict(list, {f: v for f, v in ruled.items() if side.get(f) == "test"})
        rng = random.Random(seed)
        g = sorted(gold)
        r = sorted(set(ruled) - set(gold))
        rng.shuffle(g)
        rng.shuffle(r)
        half = max_frames // 2
        chosen = g[:half] + r[:max_frames - min(half, len(g))]
        uris = {}
        for k in range(0, len(chosen), 500):
            c = [uuid.UUID(f) for f in chosen[k:k + 500]]
            uris.update({str(f): u for f, u in (await db.execute(
                select(Frame.frame_id, Frame.img_uri).where(Frame.frame_id.in_(c)))).all()})
    return chosen, uris, gold, ruled


def make_detector(name: str, champion_weights: str | None, supported: set[int], candidate: str | None = None):
    if name in ("champion", "candidate"):
        from services.autolabel.paths.path_a_detect import YoloPath
        return YoloPath(champion_weights if name == "champion" else candidate)
    if name == "path_b":
        from services.autolabel.paths.path_b_openvocab import OpenVocabSamPath
        return OpenVocabSamPath(supported_ids=supported)
    from services.autolabel.paths.path_d_grounded import GroundedPath
    return GroundedPath("grounding_dino" if name == "gdino" else "owlv2", supported_ids=supported)


def run_detector(name, det, frames, uris) -> dict[str, list]:
    import cv2
    import numpy as np

    from core.storage import get_object_store

    store = get_object_store()
    det.load()
    out: dict[str, list] = {}
    t0 = time.perf_counter()
    try:
        for n, fid in enumerate(frames):
            uri = uris.get(fid)
            if not uri:
                continue
            img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out[fid] = [{"class_id": d.class_id, "bbox": list(d.bbox), "conf": d.conf} for d in det.infer(img)]
            if n % 50 == 0:
                print(f"  {name}: {n}/{len(frames)} frames, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        det.unload()
    return out


def score(dets: dict[str, list], frames, gold, ruled, name_of) -> dict:
    g_n = g_cls = g_any = s_n = s_cls = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    r_n = r_cls = r_none = j_n = j_fire = 0
    for fid in frames:
        if fid not in dets:
            continue
        ds = dets[fid]
        for g in gold.get(fid, []):
            g_n += 1
            small = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1]) < SMALL
            hit_any = any(iou(g["bbox"], d["bbox"]) >= 0.5 for d in ds)
            hit_cls = any(iou(g["bbox"], d["bbox"]) >= 0.5 and d["class_id"] == g["class_id"] for d in ds)
            g_any += hit_any
            g_cls += hit_cls
            pc = per_class[name_of(g["class_id"])]
            pc[0] += 1
            pc[1] += hit_cls
            if small:
                s_n += 1
                s_cls += hit_cls
        for r in ruled.get(fid, []):
            best = max(ds, key=lambda d: iou(r["bbox"], d["bbox"]), default=None)
            overlap = best is not None and iou(r["bbox"], best["bbox"]) >= 0.5
            if r["not_object"]:
                j_n += 1
                j_fire += overlap
            else:
                r_n += 1
                if not overlap:
                    r_none += 1
                elif best["class_id"] == r["truth"]:
                    r_cls += 1

    def rate(k, n):
        return round(k / n, 4) if n else None

    return {"gold_boxes": g_n, "gold_recall_class": rate(g_cls, g_n), "gold_recall_any": rate(g_any, g_n),
            "small_boxes": s_n, "small_recall_class": rate(s_cls, s_n),
            "ruled_boxes": r_n, "ruled_class_correct": rate(r_cls, r_n), "ruled_missed": rate(r_none, r_n),
            "not_object_boxes": j_n, "junk_refire": rate(j_fire, j_n),
            "per_class_recall": {k: {"n": v[0], "recall": rate(v[1], v[0])}
                                 for k, v in sorted(per_class.items(), key=lambda kv: -kv[1][0])}}


def union(a: dict, b: dict) -> dict:
    return {f: a.get(f, []) + b.get(f, []) for f in set(a) | set(b)}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--cache", default=None, help="directory for cached detections")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--allow-office-hours", action="store_true")
    ap.add_argument("--test-split-only", action="store_true",
                    help="only frames from test-split blocks (needed to compare a detector retrained on the corpus)")
    ap.add_argument("--candidate", default=None, help="weights for a 'candidate' detector to compare")
    args = ap.parse_args()

    from services.tools.registry import in_quiet_hours
    if in_quiet_hours() and not args.allow_office_hours:
        print("the detector benchmark uses the GPU; it waits for office hours to end (10:00-18:00 IST)",
              file=sys.stderr)
        return 2

    from core.storage import get_object_store
    from db.session import get_sessionmaker
    from services.autolabel.grounding import supported_concept_ids
    from services.autolabel.ontology import get_ontology
    from services.labelagent.benchmark import namers

    onto = get_ontology()
    name_of, _ = namers(onto)
    frames, uris, gold, ruled = await frame_sets(args.max_frames, test_only=args.test_split_only)
    supported = await supported_concept_ids()
    from services.autolabel.runner import resolve_detector_weights

    async with get_sessionmaker()() as db:
        champion = await resolve_detector_weights(db)
    print(f"{len(frames)} frames ({sum(len(gold.get(f, [])) for f in frames)} gold boxes, "
          f"{sum(len(ruled.get(f, [])) for f in frames)} ruled boxes); champion {champion}")

    cache = Path(args.cache or Path(__file__).resolve().parents[1] / ".scratch" / "detector-benchmark")
    cache.mkdir(parents=True, exist_ok=True)
    dets: dict[str, dict] = {}
    for name in [d.strip() for d in args.detectors.split(",")]:
        tag = f"{args.max_frames}{'-test' if args.test_split_only else ''}"
        if name == "candidate":
            tag += "-" + Path(args.candidate).stem
        f = cache / f"{name}-{tag}.json"
        if f.exists():
            dets[name] = json.loads(f.read_text())
            print(f"{name}: cached ({f})")
            continue
        dets[name] = run_detector(name, make_detector(name, champion, supported, args.candidate), frames, uris)
        f.write_text(json.dumps(dets[name]))

    if "champion" in dets and "path_b" in dets:
        dets["champion+path_b (serving)"] = union(dets["champion"], dets["path_b"])
        for d in ("gdino", "owlv2"):
            if d in dets:
                dets[f"champion+path_b+{d}"] = union(dets["champion+path_b (serving)"], dets[d])
    if "candidate" in dets and "path_b" in dets:
        dets["candidate+path_b"] = union(dets["candidate"], dets["path_b"])

    report = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "frames": len(frames),
              "champion": champion, "candidate": args.candidate, "test_split_only": args.test_split_only,
              "detectors": {}}
    print(f"\n{'detector':<30} {'gold rec':>8} {'any':>6} {'small':>6} {'ruled cls':>9} {'missed':>7} {'junk':>6}")
    for name, d in dets.items():
        s = score(d, frames, gold, ruled, name_of)
        report["detectors"][name] = s

        def f(v):
            return "   -" if v is None else f"{v * 100:5.1f}%"

        print(f"{name:<30} {f(s['gold_recall_class']):>8} {f(s['gold_recall_any']):>6} {f(s['small_recall_class']):>6}"
              f" {f(s['ruled_class_correct']):>9} {f(s['ruled_missed']):>7} {f(s['junk_refire']):>6}")

    out = cache / f"report-{args.max_frames}{'-test' if args.test_split_only else ''}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    if args.save:
        store = get_object_store()
        key = f"labelagent/detectors/{report['created'].replace(':', '')}.json"
        store.put_bytes(key, json.dumps(report, indent=1).encode(), "application/json")
        store.put_bytes("labelagent/detectors/latest.json", json.dumps(report, indent=1).encode(), "application/json")
        print(f"saved {store.uri(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
