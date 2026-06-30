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
import os
from pathlib import Path

import numpy as np

# Default to the validated SegFormer-Cityscapes (clean load, verified on real frames). Mapillary Mask2Former
# is selectable but loaded with missing weights on this pod and underperformed; both are supported.
SEG_MODEL = os.environ.get("PERCEPTION_SEG_MODEL", "nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
# Substring rules over the model's own id2label -> ternary surface. Drivable surfaces a vehicle travels on;
# non-drivable walkable/curb surfaces; fallback the unpaved verge (the IDD drivable-fallback case). railroad
# is excluded from drivable.
_DRIVE = ("road", "driveway", "crosswalk", "bike lane", "bike-lane", "service lane", "service-lane", "parking")
_NONDR = ("sidewalk", "curb", "pedestrian")
_FALL = ("terrain", "pothole", "unpaved", "dirt")
_MARKING = ("lane marking", "lane-marking", "lane_marking")   # Mapillary lane-marking classes -> lanes


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


def _build_marking_ids(id2label: dict) -> list[int]:
    return [int(i) for i, lab in id2label.items() if any(k in str(lab).lower() for k in _MARKING)]


def _lanes_from_marking_mask(mask, min_pixels=80, min_height=20, n_points=8, max_lanes=8):
    """Cluster a lane-marking mask into per-line control-point polylines. Inlined mirror of the tested
    services/autolabel/lane/marking.py (perception_pod.py is scp'd standalone, so it cannot import it)."""
    import cv2
    m = np.asarray(mask).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(m)
    lanes = []
    for lab in range(1, n_labels):
        ys, xs = np.where(labels == lab)
        if len(xs) < min_pixels:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        if y1 - y0 < min_height:
            continue
        band = max(2.0, (y1 - y0) / (2.0 * n_points))
        pts = [[round(float(xs[np.abs(ys - yl) <= band].mean()), 1), round(float(yl), 1)]
               for yl in np.linspace(y0, y1, n_points) if (np.abs(ys - yl) <= band).any()]
        if len(pts) >= 2:
            lanes.append(pts)
    lanes.sort(key=lambda p: -(p[-1][1] - p[0][1]))
    return lanes[:max_lanes]


_CITY_SURFACE = {0: "drivable", 1: "non_drivable", 9: "fallback"}   # cityscapes road/sidewalk/terrain


def _load_seg():
    """Load the configured semantic segmenter. Returns (kind, proc, model, torch, surface_map, marking_ids).
    Cityscapes has no lane-marking class, so marking_ids is empty there and lanes need the Mapillary model."""
    import torch
    if "mask2former" in SEG_MODEL:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL).to("cuda").eval()
        return ("mask2former", proc, model, torch, _build_surface_map(model.config.id2label),
                _build_marking_ids(model.config.id2label))
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to("cuda").eval()
    return "segformer", proc, model, torch, _CITY_SURFACE, []


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


def _perceive(seg, pil_img, want_lanes: bool) -> dict:
    """One segmentation pass -> drivable surface (+ lanes from the lane-marking class when requested and the
    model has marking classes, i.e. Mapillary). Lanes come free from the same forward pass, no second model."""
    kind, proc, model, torch, surf, marking_ids = seg
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
    out = {"drivable": {"classes": classes, "coverage": cov, "width": w, "height": h, "model": tag}}
    if want_lanes:
        if marking_ids:
            out["lanes"] = _lanes_from_marking_mask(np.isin(labels, marking_ids))
        else:
            out["lanes"], out["lane_error"] = [], "no lane-marking class in this model (use a Mapillary model)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", action="store_true", help="also derive lanes from the lane-marking class")
    args = ap.parse_args()

    from PIL import Image
    seg = _load_seg()
    frames = [json.loads(ln) for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    print(f"[perception] {len(frames)} frames, lanes={'on' if args.lanes else 'off'}, model={SEG_MODEL}")
    with open(args.out, "w") as fout:
        for i, fr in enumerate(frames):
            rec: dict = {"frame_id": fr["frame_id"]}
            try:
                rec.update(_perceive(seg, Image.open(fr["path"]).convert("RGB"), args.lanes))
            except Exception as exc:  # noqa: BLE001  one bad frame must not abort the sweep
                rec["drivable_error"] = str(exc)[:160]
            fout.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                print(f"[perception] {i + 1}/{len(frames)}")
    print(f"[perception] done -> {args.out}")


if __name__ == "__main__":
    main()
