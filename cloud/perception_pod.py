"""Pod-side perception runtime (drivable + lanes), the real models the local fallbacks stand in for.

Drivable: Mask2Former fine-tuned on Mapillary Vistas -> a dense semantic segmentation whose road / sidewalk /
terrain classes ARE the ternary drivable surface (drivable / non_drivable / fallback). Mapillary is globally
diverse (developing-country roads included), so it covers Indian dashcam roads far better than Cityscapes,
which under-segmented them. SAM 3.1 PCS was ruled out first: it segments object concepts, not stuff ("road"
returns nothing). The surface mapping is built from the model's own id2label, so it adapts to the label set.
Lanes: CLRerNet (mmdet) -> per-lane point lists, guarded so a lane failure never blocks drivable.

Reads a manifest (one JSON object per line: {frame_id, path}) and writes perception.jsonl (one object per
line: {frame_id, drivable:{classes,coverage,width,height}, lanes:[[ [x,y], ... ]], lane_error?}). The local
side (services/perception/cloud.py) pushes the frames + manifest, runs this, pulls the result, and ingests
into DrivableMask + Lane. No em-dashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import os

# Default to the validated SegFormer-Cityscapes (clean load, verified on real frames). Mapillary Mask2Former
# is selectable but loaded with missing weights on this pod and underperformed; both are supported.
SEG_MODEL = os.environ.get("PERCEPTION_SEG_MODEL", "nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
# Substring rules over the model's own id2label -> ternary surface. Drivable surfaces a vehicle travels on;
# non-drivable walkable/curb surfaces; fallback the unpaved verge (the IDD drivable-fallback case). railroad
# is excluded from drivable.
_DRIVE = ("road", "driveway", "crosswalk", "bike lane", "bike-lane", "service lane", "service-lane", "parking")
_NONDR = ("sidewalk", "curb", "pedestrian")
_FALL = ("terrain", "pothole", "unpaved", "dirt")


def _build_surface_map(id2label: dict) -> dict:
    out = {}
    for i, lab in id2label.items():
        low = str(lab).lower()
        if "rail" in low:
            continue
        if any(k in low for k in _DRIVE):
            out[int(i)] = "drivable"
        elif any(k in low for k in _NONDR):
            out[int(i)] = "non_drivable"
        elif any(k in low for k in _FALL):
            out[int(i)] = "fallback"
    return out


_CITY_SURFACE = {0: "drivable", 1: "non_drivable", 9: "fallback"}   # cityscapes road/sidewalk/terrain


def _load_seg():
    """Load the configured semantic segmenter. Returns (kind, proc, model, torch, surface_map)."""
    import torch
    if "mask2former" in SEG_MODEL:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL).to("cuda").eval()
        return "mask2former", proc, model, torch, _build_surface_map(model.config.id2label)
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to("cuda").eval()
    return "segformer", proc, model, torch, _CITY_SURFACE


def _mask_to_polygons(mask) -> list[list[float]]:
    """Boolean mask -> list of flat [x1,y1,x2,y2,...] polygons (the DrivableMask JSON shape)."""
    import cv2
    m = np.asarray(mask).astype(np.uint8)
    if m.ndim == 3:
        m = m[0]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        eps = 0.004 * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            polys.append([round(float(x), 1) for xy in ap for x in xy])
    return polys


def _drivable_for(seg, pil_img) -> dict:
    kind, proc, model, torch, surf = seg
    w, h = pil_img.size
    inputs = proc(images=pil_img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)
    if kind == "mask2former":
        labels = proc.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0].cpu().numpy()
        tag = "mask2former-mapillary:pod"
    else:
        up = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)[0].cpu().numpy()
        tag = "segformer-cityscapes:pod"
    classes: dict = {"drivable": [], "non_drivable": [], "fallback": []}
    cov = {}
    total = float(w * h) or 1.0
    for cls in classes:
        ids = [i for i, c in surf.items() if c == cls]
        m = np.isin(labels, ids) if ids else np.zeros_like(labels, bool)
        classes[cls] = _mask_to_polygons(m)
        cov[cls] = round(float(m.sum()) / total, 4)
    return {"classes": classes, "coverage": cov, "width": w, "height": h, "model": tag}


def _load_clrernet():
    """CLRerNet via mmdet. Returns (infer_fn, tag) or raises so the caller can guard lanes."""
    from mmdet.apis import inference_detector, init_detector
    import os
    cfg = os.environ.get("CLRERNET_CONFIG", "/workspace/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    ckpt = os.environ.get("CLRERNET_CKPT", "/workspace/ckpts/clrernet_culane_dla34_ema.pth")
    model = init_detector(cfg, ckpt, device="cuda:0")

    def infer(path: str) -> list[list[list[float]]]:
        res = inference_detector(model, path)
        lanes = getattr(res, "pred_instances", None)
        out = []
        # CLRerNet returns lane point-lists; the demo exposes result.pred_instances.lanes or .scores+.lanes
        raw = getattr(lanes, "lanes", None) if lanes is not None else None
        if raw is None:
            raw = res if isinstance(res, list) else []
        for lane in raw:
            pts = [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(lane).reshape(-1, 2)]
            if len(pts) >= 2:
                out.append(pts)
        return out
    return infer, "clrernet:pod"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", action="store_true", help="also run CLRerNet lane proposals")
    args = ap.parse_args()

    from PIL import Image
    seg = _load_seg()
    lane_infer = lane_err = None
    if args.lanes:
        try:
            lane_infer, _ = _load_clrernet()
        except Exception as exc:  # noqa: BLE001  lanes must never block drivable
            lane_err = f"CLRerNet unavailable: {str(exc)[:160]}"
            print(f"[perception] {lane_err}")

    frames = [json.loads(ln) for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    print(f"[perception] {len(frames)} frames, lanes={'on' if lane_infer else ('error' if args.lanes else 'off')}")
    with open(args.out, "w") as fout:
        for i, fr in enumerate(frames):
            rec: dict = {"frame_id": fr["frame_id"]}
            try:
                rec["drivable"] = _drivable_for(seg, Image.open(fr["path"]).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                rec["drivable_error"] = str(exc)[:160]
            if lane_infer is not None:
                try:
                    rec["lanes"] = lane_infer(fr["path"])
                except Exception as exc:  # noqa: BLE001
                    rec["lanes"], rec["lane_error"] = [], str(exc)[:160]
            elif lane_err:
                rec["lanes"], rec["lane_error"] = [], lane_err
            fout.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                print(f"[perception] {i + 1}/{len(frames)}")
    print(f"[perception] done -> {args.out}")


if __name__ == "__main__":
    main()
