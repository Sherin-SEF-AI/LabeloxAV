"""Heavy relabel re-inference entrypoint - runs ON the RunPod A100 (not locally). Loads the champion
weights and re-infers a manifest of frames, matching detections to the existing objects by IoU and
emitting a new class + confidence per object. The local side ingests relabeled.jsonl through the diff +
apply path (services.relabel.run.ingest_model_relabel), which auto-applies safe improvements, never
touches human-verified objects, and lands on a new lakeFS branch.

  python cloud/relabel_pod.py --weights /workspace/in/champion.pt \
      --manifest /workspace/in/manifest.jsonl --out /workspace/out/relabeled.jsonl

manifest.jsonl: one JSON object per line:
  {"frame_id", "path": "/workspace/in/<file>.jpg",
   "objects": [{"object_id", "bbox": [x1,y1,x2,y2]}]}
relabeled.jsonl: one JSON object per line, one per matched object:
  {"object_id", "class_name", "conf"}

No em-dashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _load_yolo(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    model = _load_yolo(args.weights)
    names = model.names
    frames = [json.loads(ln) for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    with open(args.out, "w") as fout:
        for fr in frames:
            res = model.predict(fr["path"], device=0, verbose=False)
            dets = []
            for r in res:
                for b in r.boxes:
                    dets.append(([float(v) for v in b.xyxy[0].tolist()], names[int(b.cls[0])], float(b.conf[0])))
            for obj in fr.get("objects", []):
                best, best_iou = None, args.iou
                for bbox, label, conf in dets:
                    iou = _iou(obj["bbox"], bbox)
                    if iou >= best_iou:
                        best, best_iou = (label, conf), iou
                if best is not None:
                    fout.write(json.dumps({"object_id": obj["object_id"], "class_name": best[0], "conf": best[1]}) + "\n")
    print(f"relabeled {len(frames)} frames from {args.weights}")


if __name__ == "__main__":
    main()
