"""Heavy autolabel entrypoint - runs ON the RunPod A100 (not locally). Reuses the smoke-test's PROVEN
model loaders (YOLO26 + SAM 3.1 PCS + Qwen3-VL) to label a manifest of frames and emit labels.jsonl,
which the local side ingests into Postgres/MinIO via services.autolabel.persist.

  python cloud/autolabel_pod.py --manifest /workspace/in/manifest.jsonl --out /workspace/out/labels.jsonl

manifest.jsonl: one JSON object per line: {"frame_id": "<uuid>", "path": "/workspace/in/<file>.jpg"}
labels.jsonl:   one JSON object per line: {"frame_id", "objects": [{"label","bbox":[x1,y1,x2,y2],
                "score","mask": [[x,y,...]]?}]}

Pipeline per frame: YOLO26 detect -> SAM 3.1 mask per detection (concept = the YOLO label) -> optional
Qwen3-VL class verification. Models load once; frames stream. No em-dashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_yolo():
    from ultralytics import YOLO

    for w in ("yolo26x.pt", "yolo26l.pt", "yolo26n.pt", "yolo11x.pt"):
        try:
            return YOLO(w), w
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no YOLO weights could be loaded on the pod")


def _load_sam():
    # Proven entry from the smoke test (PASS): build_sam3_image_model(bpe_path=...) + Sam3Processor.
    import sam3.model_builder as _mb
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = str(Path(_mb.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    return Sam3Processor(build_sam3_image_model(bpe_path=bpe))


def _mask_to_polygon(mask) -> list[list[float]]:
    import cv2
    import numpy as np

    m = (np.asarray(mask) > 0.5).astype("uint8")
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in c[::2]]  # decimate every other point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", action="store_true", help="add SAM 3.1 masks per detection")
    args = ap.parse_args()

    import torch

    yolo, wname = _load_yolo()
    sam = _load_sam() if args.masks else None
    print(f"loaded detector={wname} masks={'sam3.1' if sam else 'off'}")

    frames = [json.loads(ln) for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_obj = 0
    with open(args.out, "w") as fout:
        for i, fr in enumerate(frames):
            path = fr["path"]
            objects = []
            res = yolo.predict(path, device=0, verbose=False)
            for r in res:
                names = r.names
                for b in r.boxes:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    label = names[int(b.cls[0])]
                    obj = {"label": label, "bbox": xyxy, "score": float(b.conf[0])}
                    if sam is not None:
                        try:
                            from PIL import Image

                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                                state = sam.set_image(Image.open(path).convert("RGB"))
                                out = sam.set_text_prompt(state=state, prompt=label)
                            if out["masks"] is not None and len(out["masks"]):
                                poly = _mask_to_polygon(out["masks"][0])
                                if poly:
                                    obj["mask"] = poly
                        except Exception as exc:  # noqa: BLE001
                            obj["mask_error"] = str(exc)[:120]
                    objects.append(obj)
                    n_obj += 1
            fout.write(json.dumps({"frame_id": fr["frame_id"], "objects": objects}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(frames)} frames, {n_obj} objects")
    print(f"DONE {len(frames)} frames, {n_obj} objects -> {args.out}")


if __name__ == "__main__":
    main()
