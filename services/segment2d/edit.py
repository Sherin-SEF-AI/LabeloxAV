"""Human editing of a dense semantic raster.

`segment_frame` produces a proposal and `FrameSegmentation.source` has always had a `human` value it could
take, but nothing could ever set it: there was no write path, so the dense layer was permanently machine
output that a person could look at and not correct. A label nobody can fix is not a label, it is a
visualisation, and it is why the table holds six rows.

Editing arrives as polygons per class rather than as a raster. That is what a canvas can produce and what a
review can diff, and it is the same shape the instance mask editor in `services/segment2d/assist.py` already
round-trips. The raster is regenerated here so the stored artefact stays exactly what every consumer already
reads: a class-id-per-pixel npz plus a coloured overlay.

Paint order is the order the polygons arrive in, later over earlier. Sorting by area or class would be
tidier and wrong: an annotator who draws a car over a road expects the car, and a rule that decides
occlusion for them is a rule they cannot see and cannot override.
"""

from __future__ import annotations

import io

import numpy as np

from core.logging import get_logger

log = get_logger("segment_edit")

MODEL_VERSION = "human"


def rasterize_class_polygons(class_polygons: list[dict], width: int, height: int,
                             *, name_to_id) -> tuple[np.ndarray, list[str]]:
    """Paint per-class polygons into a class-id raster.

    class_polygons is [{class_name, polygons: [[x,y,x,y,...], ...]}]. Unknown class names are collected and
    returned rather than raising: an edit that names one class the ontology dropped should land the rest of
    the work, and the caller decides whether to tell the annotator or refuse.
    """
    import cv2

    labels = np.zeros((height, width), dtype=np.int32)
    unknown: list[str] = []
    for entry in class_polygons:
        name = str(entry.get("class_name") or "")
        cid = name_to_id(name)
        if cid is None:
            unknown.append(name)
            continue
        rings = []
        for flat in (entry.get("polygons") or []):
            pts = [float(v) for v in flat]
            if len(pts) < 6:
                # Fewer than three points is not a region. Skipped rather than closed into a degenerate
                # sliver, which would put a one-pixel class into the coverage stats.
                continue
            ring = np.array(pts, dtype=np.float64).reshape(-1, 2)
            ring[:, 0] = np.clip(ring[:, 0], 0, width - 1)
            ring[:, 1] = np.clip(ring[:, 1], 0, height - 1)
            rings.append(np.round(ring).astype(np.int32))
        if rings:
            cv2.fillPoly(labels, rings, int(cid))
    return labels, unknown


def coverage_of(labels: np.ndarray, id_to_name) -> dict[str, float]:
    """Pixel fraction per class. Class 0 is unlabelled and is not a class."""
    total = float(labels.size) or 1.0
    out: dict[str, float] = {}
    for cid in np.unique(labels):
        if int(cid) == 0:
            continue
        name = id_to_name(int(cid))
        if name is None:
            continue
        out[name] = round(float(np.count_nonzero(labels == cid)) / total, 4)
    return out


def overlay_rgba(labels: np.ndarray) -> np.ndarray:
    """The coloured display raster, in the same BGRA convention the proposer writes."""
    from services.segment2d.semantic import _class_color

    h, w = labels.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cid in np.unique(labels):
        if int(cid) == 0:
            continue
        r, g, b = _class_color(int(cid))
        overlay[labels == cid] = (b, g, r, 140)
    return overlay


def store_labels(store, session_id, frame_id, kind: str, labels: np.ndarray) -> tuple[str, str | None]:
    """Write the raster and its overlay, returning both uris.

    Written under a new random key rather than overwriting the previous one so a rollback has something to
    roll back to; the row points at the current pair and the old objects are collected by lifecycle policy.
    """
    import cv2

    from services.segment2d.semantic import _rand

    key = f"segmentation/{session_id}/{frame_id}/{kind}"
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=labels)
    labels_uri = store.put_bytes(f"{key}/labels/{_rand()}.npz", buf.getvalue(),
                                 "application/octet-stream")
    ok, png = cv2.imencode(".png", overlay_rgba(labels))
    overlay_uri = (store.put_bytes(f"{key}/overlay/{_rand()}.png", png.tobytes(), "image/png")
                   if ok else None)
    return labels_uri, overlay_uri


def merge_onto_existing(base: np.ndarray | None, edit: np.ndarray) -> np.ndarray:
    """Lay an edit over an existing raster, keeping what the edit did not touch.

    Without this, editing one region would erase every other region on the frame, because the canvas only
    ever sends back what the annotator drew. Zero in the edit means "not mentioned", not "erase": erasing is
    a separate operation and has to be, or an annotator correcting a car would silently delete the road.
    """
    if base is None or base.shape != edit.shape:
        return edit
    out = base.copy()
    touched = edit != 0
    out[touched] = edit[touched]
    return out
