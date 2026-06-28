"""An AB3DMOT-style 3D multi-object tracker: a constant-velocity Kalman filter per object, associated frame
to frame by 3D IoU (services/lidar/boxes.iou_3d) with the Hungarian algorithm, optionally weighted by the
DINOv3 appearance of the projected 2D crop. Each 3D track links to the M2.0 2D track: a lifted detection
carries its 2D object's track_id, so the 3D and 2D tracks are the same physical object; appearance breaks
ties and links native detections that have no 2D track yet.

Local and light (Principle: interactive work stays on the box). No new heavy dependency: the Kalman filter is
plain numpy and the assignment is scipy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.config import get_settings
from core.logging import get_logger
from services.lidar.boxes import iou_3d

log = get_logger("lidar_track3d")


class KalmanBoxTracker3D:
    """Constant-velocity Kalman filter on the box centre [x, y, z, vx, vy, vz]; dims and yaw carry the latest
    measurement, lightly smoothed. The box is in the ego frame, metres."""

    _next_id = 0

    def __init__(self, det: dict, dt: float):
        cx, cy, cz = det["center"]
        self.x = np.array([cx, cy, cz, 0.0, 0.0, 0.0], dtype=np.float64)
        self.p = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
        self.dims = list(det["dims"])
        self.yaw = float(det["yaw"])
        self.class_id = det.get("class_id")
        self.id = KalmanBoxTracker3D._next_id
        KalmanBoxTracker3D._next_id += 1
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.track2d_votes: Counter = Counter()
        if det.get("track_id_2d"):
            self.track2d_votes[str(det["track_id_2d"])] += 1
        self.appearance = det.get("appearance")
        self.members: list[dict] = [det]
        self.history: list[dict] = [self._box()]

    def _box(self) -> dict:
        return {"center": [float(self.x[0]), float(self.x[1]), float(self.x[2])],
                "dims": [float(d) for d in self.dims], "yaw": float(self.yaw)}

    def predict(self, dt: float) -> dict:
        f = np.eye(6)
        f[0, 3] = f[1, 4] = f[2, 5] = dt
        q = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.age += 1
        self.time_since_update += 1
        return self._box()

    def update(self, det: dict) -> None:
        z = np.array(det["center"], dtype=np.float64)
        h = np.zeros((3, 6))
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        r = np.diag([0.3, 0.3, 0.3])
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(6) - k @ h) @ self.p
        self.dims = [0.7 * a + 0.3 * b for a, b in zip(self.dims, det["dims"], strict=False)]
        self.yaw = float(det["yaw"])
        self.hits += 1
        self.time_since_update = 0
        if det.get("track_id_2d"):
            self.track2d_votes[str(det["track_id_2d"])] += 1
        if det.get("appearance") is not None:
            ap = np.asarray(det["appearance"], dtype=np.float32)
            self.appearance = ap if self.appearance is None else 0.7 * np.asarray(self.appearance) + 0.3 * ap
        self.members.append(det)
        self.history.append(self._box())

    @property
    def velocity(self) -> list[float]:
        return [float(self.x[3]), float(self.x[4]), float(self.x[5])]

    def linked_track_2d(self) -> str | None:
        return self.track2d_votes.most_common(1)[0][0] if self.track2d_votes else None


def _appearance_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0


class Tracker3D:
    """Manages the set of 3D tracks across frames."""

    def __init__(self, iou_thresh: float | None = None, max_age: int | None = None,
                 min_hits: int | None = None, appearance_w: float | None = None):
        cfg = get_settings().lidar
        self.iou_thresh = iou_thresh if iou_thresh is not None else cfg.track3d_iou_thresh
        self.max_age = max_age if max_age is not None else cfg.track3d_max_age
        self.min_hits = min_hits if min_hits is not None else cfg.track3d_min_hits
        self.appearance_w = appearance_w if appearance_w is not None else cfg.track3d_appearance_w
        self.tracks: list[KalmanBoxTracker3D] = []
        self.retired: list[KalmanBoxTracker3D] = []

    def step(self, detections: list[dict], dt: float = 0.1) -> list[dict]:
        """Advance one frame. Returns, for each detection, the track id it was assigned to."""
        predicted = [t.predict(dt) for t in self.tracks]
        assignments: dict[int, int] = {}
        if self.tracks and detections:
            cost = np.ones((len(self.tracks), len(detections)), dtype=np.float64)
            iou = np.zeros_like(cost)
            for i, pbox in enumerate(predicted):
                for j, det in enumerate(detections):
                    iou[i, j] = iou_3d(pbox, det)
                    appear = _appearance_sim(self.tracks[i].appearance, det.get("appearance"))
                    cost[i, j] = (1.0 - iou[i, j]) + self.appearance_w * (1.0 - appear)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=False):
                if iou[r, c] >= self.iou_thresh:
                    self.tracks[r].update(detections[c])
                    assignments[c] = self.tracks[r].id

        for j, det in enumerate(detections):
            if j not in assignments:
                t = KalmanBoxTracker3D(det, dt)
                self.tracks.append(t)
                assignments[j] = t.id

        alive = [t for t in self.tracks if t.time_since_update <= self.max_age]
        self.retired.extend(t for t in self.tracks if t.time_since_update > self.max_age)
        self.tracks = alive
        out = []
        for j, det in enumerate(detections):
            out.append({"detection_index": j, "track_3d_local_id": assignments.get(j),
                        "object_3d_id": det.get("object_3d_id")})
        return out

    def all_tracks(self) -> list[KalmanBoxTracker3D]:
        """Every track ever created (alive plus retired), for finalizing a session's track_3d rows."""
        return self.tracks + self.retired

    def confirmed(self) -> list[KalmanBoxTracker3D]:
        return [t for t in self.all_tracks() if t.hits >= self.min_hits]
