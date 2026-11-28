"""Add fully labelled IDD Segmentation *train* frames to a ruled corpus training set, so a retrain learns the
corpus's rulings without forgetting what the champion knew.

    .venv/bin/python scripts/build_idd_mix.py --base .scratch/training/ruled-v1 --out .scratch/training/ruled-v2 --n 5000

Why. ruled-v1 (corpus only, fine-tuned from the IDD champion) got the corpus's hard cases right more often
(class on ruled boxes 28% -> 44%, re-fire on not-an-object boxes 86% -> 58%) but lost a sixth of its recall
on IDD (42% -> 35%): the corpus labels a few objects per frame, so the rest of the traffic it shows is
weakly taught, and the IDD knowledge washed out. IDD frames label every instance, which is the cure.

Only IDD's *train* split is used. Its val split is what scripts/idd_pipeline_eval.py scores on, so the
accuracy number stays clean.

IDD categories map onto the base set's class names (`car` is `sedan`, as the champion was trained). Poles
and structures stay background, which matches the corpus, where nobody boxes them either. The base set's
val split is kept as the only val split, so training is judged on the corpus, and IDD is judged afterwards
by the pipeline eval.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

IDD = Path("/home/jo/lake/idd/segmentation")
TO_CLASS = {"car": "sedan", "motorcycle": "motorcycle", "rider": "rider", "person": "pedestrian",
            "autorickshaw": "autorickshaw", "truck": "truck", "bus": "bus", "bicycle": "cycle", "animal": "cattle",
            "traffic sign": "traffic_sign", "traffic light": "traffic_signal", "billboard": "hoarding",
            "vehicle fallback": "vehicle_fallback", "caravan": "vehicle_fallback", "trailer": "vehicle_fallback"}


def names_of(base: Path) -> list[str]:
    names, on = [], False
    for line in (base / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            on = True
        elif on and ":" in line:
            names.append(line.split(":", 1)[1].strip())
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=".scratch/training/ruled-v1")
    ap.add_argument("--out", default=".scratch/training/ruled-v2")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    base, out = Path(args.base).resolve(), Path(args.out).resolve()
    names = names_of(base)
    idx = {n: i for i, n in enumerate(names)}
    polys = sorted(glob.glob(str(IDD / "gtFine" / "train" / "*" / "*_gtFine_polygons.json")))
    random.Random(args.seed).shuffle(polys)
    img_dir, lab_dir = out / "idd" / "images" / "train", out / "idd" / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    kept = boxes = 0
    for p in map(Path, polys):
        if kept >= args.n:
            break
        stem = p.name.replace("_gtFine_polygons.json", "")
        img = IDD / "leftImg8bit" / "train" / p.parent.name / f"{stem}_leftImg8bit.png"
        if not img.exists():
            continue
        j = json.loads(p.read_text())
        w, h = float(j["imgWidth"]), float(j["imgHeight"])
        lines = []
        for o in j["objects"]:
            cls = TO_CLASS.get(o.get("label", ""))
            pts = o.get("polygon") or []
            if cls is None or o.get("deleted") or len(pts) < 3:
                continue
            xs = [min(max(q[0], 0.0), w) for q in pts]
            ys = [min(max(q[1], 0.0), h) for q in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            lines.append(f"{idx[cls]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} "
                         f"{(y2 - y1) / h:.6f}")
        name = f"idd_{p.parent.name}_{stem}"
        link = img_dir / f"{name}.png"
        if not link.exists():
            link.symlink_to(img)
        (lab_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        kept += 1
        boxes += len(lines)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain:\n  - {base / 'images' / 'train'}\n  - {img_dir}\nval: {base / 'images' / 'val'}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "build.json").write_text(json.dumps({"base": str(base), "idd_frames": kept, "idd_boxes": boxes,
                                                "args": vars(args)}, indent=1))
    print(f"{kept} IDD train frames, {boxes} boxes -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
