"""Dense full-frame segmentation. SAM-everything proposes regions, the VLM classifies each, and the
regions are stitched into a per-pixel class-id raster (semantic) plus a per-pixel instance-id raster
(panoptic). Reuses the shared RegionAdapter and VlmClassifyAdapter (no second model copy). The rasters
and a colored display overlay are stored in the object store. Heavy imports are lazy so this module
loads without numpy/cv2/torch.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("segment2d")
MODEL_VERSION = "sam-everything+vlm-0.1"


def _class_color(cid: int) -> tuple[int, int, int]:
    """A deterministic RGB per class id, so the overlay is stable across runs."""
    return ((cid * 67 + 41) % 256, (cid * 131 + 17) % 256, (cid * 193 + 89) % 256)


def segment_frame(image_bgr, frame_id, session_id, store, onto, settings, *, backend=None,
                  kind: str = "semantic") -> dict:
    """Produce the dense rasters for a frame and store them. Returns the uris, coverage, and the panoptic
    segments map (instance_id -> {class_id}). object linking is done by the caller for panoptic."""
    import cv2
    import numpy as np

    cfg = settings.phase4.recall
    be = backend
    if be is None:
        from services.recall.backends import build_backends
        be = build_backends(settings)

    h, w = image_bgr.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)        # 0 = background/unlabeled
    instance = np.zeros((h, w), dtype=np.int32)
    segments: dict[str, dict] = {}

    regions = be.region_adapter.propose_masks(image_bgr)
    # paint largest first so smaller regions land on top (a sign over a building, a rider over a bike)
    regions.sort(key=lambda mb: -int(np.count_nonzero(mb[0])))
    inst = 0
    for mask, bbox in regions:
        name, conf = be.classify(image_bgr, bbox)
        if name is None or conf < cfg.region_min_vlm_conf or not onto.has_name(name):
            continue
        cid = onto.by_name(name).id
        m = mask if mask.shape == (h, w) else cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)
        labels[m] = cid
        inst += 1
        instance[m] = inst
        segments[str(inst)] = {"class_id": cid, "class_name": name, "object_id": None}
    log.info("segment2d.frame", kind=kind, regions=len(regions), instances=inst)

    if kind == "panoptic":
        # Panoptic segments must tile the frame without overlap, so each one's polygon has to follow the
        # visible edge precisely and exclude whatever occludes it. The instance raster already encodes
        # that: regions were painted largest-first, so a vehicle in front of a retaining wall overwrote
        # the wall's pixels there. Polygonize each segment straight from that raster with a fixed pixel
        # tolerance (so a large stuff region is no coarser than a small thing) and keep interior holes, so
        # a stuff region keeps the cut-out where an occluding vehicle sits instead of swallowing it.
        # (Semantic stays raster-only; a coarse boundary is acceptable there.)
        from services.autolabel.paths.path_b_sam3 import polygons_from_mask
        for sid, seg in segments.items():
            seg["polygon"] = polygons_from_mask(instance == int(sid), keep_holes=True, epsilon_px=1.5)

    total = float(h * w) or 1.0
    coverage: dict[str, float] = {}
    for cid in np.unique(labels):
        if cid == 0:
            continue
        coverage[onto.by_id(int(cid)).name] = round(float(np.count_nonzero(labels == cid)) / total, 4)

    # colored RGBA overlay for display: each labelled pixel gets its class colour at partial alpha
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cid in np.unique(labels):
        if cid == 0:
            continue
        r, g, b = _class_color(int(cid))
        sel = labels == cid
        overlay[sel] = (b, g, r, 140)  # BGRA for cv2 png encode

    key = f"segmentation/{session_id}/{frame_id}/{kind}"
    labels_uri = _put_npz(store, f"{key}/labels", labels)
    ov_ok, ov_buf = cv2.imencode(".png", overlay)
    overlay_uri = (store.put_bytes(f"{key}/overlay/{_rand()}.png", ov_buf.tobytes(), "image/png")
                   if ov_ok else None)
    instance_uri = _put_npz(store, f"{key}/instance", instance) if kind == "panoptic" else None

    return {"labels_uri": labels_uri, "instance_uri": instance_uri, "overlay_uri": overlay_uri,
            "coverage": coverage, "segments": segments, "n_instances": inst,
            "model_version": MODEL_VERSION, "width": w, "height": h}


def _rand() -> str:
    import uuid
    return uuid.uuid4().hex


def _put_npz(store, key_prefix: str, arr) -> str:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr)
    return store.put_bytes(f"{key_prefix}/{_rand()}.npz", buf.getvalue(), "application/octet-stream")
