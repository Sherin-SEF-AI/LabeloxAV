"""The one internal point cloud representation every source normalizes to.

A Cloud is xyz plus intensity plus a PPS timestamp, with an optional ring or return index. Real LiDAR,
pseudo-LiDAR (camera depth lift), and public datasets all produce this same shape, so every later stage
(clean, viewer, projection, calibration) is source agnostic. Time is UTC nanoseconds, integer, on the PPS
base, so a cloud and the camera frames captured at the same ts_ns are one query. The canonical on-disk form
is a compressed .npz, lossless and compact, written to the object store.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Cloud:
    xyz: np.ndarray                       # (N, 3) float32, metres in `frame`
    intensity: np.ndarray                 # (N,) float32, normalized to 0..1 where the source allows
    ts_ns: int                            # UTC nanoseconds (PPS base); the cloud identity
    ring: np.ndarray | None = None        # (N,) int16 laser ring / return index, when the source provides it
    source: str = "lidar"                 # lidar | pseudo | dataset
    frame: str = "ego"                    # coordinate frame label (kitti_velo, nuscenes_lidar, ego, world)
    depth_model: str | None = None        # pinned checkpoint, for pseudo-LiDAR clouds
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        n = self.xyz.shape[0]
        if self.intensity is None or len(self.intensity) != n:
            self.intensity = np.zeros(n, dtype=np.float32)
        else:
            self.intensity = np.ascontiguousarray(self.intensity, dtype=np.float32).reshape(-1)
        if self.ring is not None:
            self.ring = np.ascontiguousarray(self.ring, dtype=np.int16).reshape(-1)
            if len(self.ring) != n:
                self.ring = None
        self.ts_ns = int(self.ts_ns)

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def bounds(self) -> dict:
        """3D extent, the value stored on the point_cloud row for fast viewport framing."""
        if self.n == 0:
            return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "n": 0}
        lo = self.xyz.min(axis=0)
        hi = self.xyz.max(axis=0)
        return {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi], "n": self.n}

    def _meta(self) -> dict:
        return {"ts_ns": self.ts_ns, "source": self.source, "frame": self.frame,
                "depth_model": self.depth_model, "calibration_version": self.calibration_version,
                "has_ring": self.ring is not None}

    def to_npz_bytes(self) -> bytes:
        """Lossless, compressed serialization for the object store. Metadata rides in a json sidecar array."""
        buf = io.BytesIO()
        arrays = {"xyz": self.xyz, "intensity": self.intensity,
                  "meta": np.frombuffer(json.dumps(self._meta()).encode("utf-8"), dtype=np.uint8)}
        if self.ring is not None:
            arrays["ring"] = self.ring
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> Cloud:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8")) if "meta" in z else {}
            ring = z["ring"] if "ring" in z.files else None
            return cls(xyz=z["xyz"], intensity=z["intensity"], ts_ns=int(meta.get("ts_ns", 0)),
                       ring=ring, source=meta.get("source", "lidar"), frame=meta.get("frame", "ego"),
                       depth_model=meta.get("depth_model"), calibration_version=meta.get("calibration_version"))

    def decimate(self, max_points: int, seed: int = 0) -> Cloud:
        """A uniform random subsample for interactive rendering. Deterministic given the seed."""
        if self.n <= max_points or max_points <= 0:
            return self
        idx = np.random.default_rng(seed).choice(self.n, size=max_points, replace=False)
        idx.sort()
        return Cloud(xyz=self.xyz[idx], intensity=self.intensity[idx], ts_ns=self.ts_ns,
                     ring=self.ring[idx] if self.ring is not None else None, source=self.source,
                     frame=self.frame, depth_model=self.depth_model,
                     calibration_version=self.calibration_version)
