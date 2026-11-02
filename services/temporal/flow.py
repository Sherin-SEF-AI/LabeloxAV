"""Optical-flow box tracking: the one step every propagator uses.

Lucas-Kanade flow on features inside the box, a RANSAC similarity fit, the box corners moved by it. Three copies
of this lived in the editor's forward propagation, the SAM propagation and the agent's propagation walk; this is
the one that guards against what the others did not: frames of different sizes (imported imagery), and a box
that blows up or collapses between adjacent frames (flow onto a featureless frame), which a real object's box
does not do.

Callers pass frames of one camera. Flow from one view into another walks the box onto something else, so
every frame query feeding this filters on cam_id.
"""

from __future__ import annotations

import cv2
import numpy as np


def gray(store, uri: str) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clip_box(b, w, h):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))))
    y1, y2 = sorted((max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))))
    return [x1, y1, x2, y2]


_LK = {"winSize": (21, 21), "maxLevel": 3,
       "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)}


def flow_box(prev_gray: np.ndarray, cur_gray: np.ndarray, box: list[float]) -> list[float] | None:
    """Move a box from prev_gray to cur_gray via LK optical flow on its interior features plus a RANSAC
    similarity fit. None when it cannot be tracked (too small, too few features, no transform)."""
    ph, pw = prev_gray.shape
    ch, cw = cur_gray.shape
    # Optical flow requires both frames to be the same size; a session with varying image dimensions (common
    # in imported imagery) otherwise crashes calcOpticalFlowPyrLK. Track in the previous frame's space by
    # resizing the current frame to match, then scale the result back to the current frame's real size.
    if (ch, cw) != (ph, pw):
        cur_gray = cv2.resize(cur_gray, (pw, ph))
    h, w = ph, pw
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = prev_gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=4)
    if pts is None or len(pts) < 4:
        return None
    pts = pts.reshape(-1, 2) + np.float32([x1, y1])
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **_LK)
    st = st.reshape(-1).astype(bool)
    if st.sum() < 4:
        return None
    m, _ = cv2.estimateAffinePartial2D(pts[st], nxt.reshape(-1, 2)[st], method=cv2.RANSAC)
    if m is None:
        return None
    corners = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    moved = cv2.transform(corners.reshape(-1, 1, 2), m).reshape(-1, 2)
    nb = clip_box([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()], w, h)
    if nb[2] - nb[0] < 4 or nb[3] - nb[1] < 4:
        return None
    # reject a blown-up or collapsed track (e.g. flow onto a featureless frame): a real object's box does not
    # jump more than ~4x in area between adjacent frames.
    old_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    if not (0.25 <= (nb[2] - nb[0]) * (nb[3] - nb[1]) / old_area <= 4.0):
        return None
    # nb is in the previous frame's space; scale it back to the current frame's real dimensions when they differ
    if (ch, cw) != (ph, pw):
        sx, sy = cw / pw, ch / ph
        nb = [nb[0] * sx, nb[1] * sy, nb[2] * sx, nb[3] * sy]
    return nb


def flow_track(grays: list, box0: list[float]) -> list[list[float] | None]:
    """Track a box across a sequence of gray frames (grays[0] is the anchor). Returns the propagated box for
    each of grays[1:], stopping (the rest None) once tracking fails."""
    out: list[list[float] | None] = []
    box, prev = list(box0), grays[0]
    for cur in grays[1:]:
        nb = flow_box(prev, cur, box) if (prev is not None and cur is not None) else None
        out.append(nb)
        if nb is None:
            break
        box, prev = nb, cur
    return out


__all__ = ["clip_box", "flow_box", "flow_track", "gray"]
