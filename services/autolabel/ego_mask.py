"""Ego-vehicle (hood) mask, auto-estimated per camera.

The car's own hood sits in the same pixels of every frame from a given camera, so it has near-zero temporal
variance and is anchored to the bottom edge. We estimate it once per (vehicle, camera) by stacking a sample
of that camera's frames, finding the low-variance pixels, and keeping the bottom-anchored connected region
(so the static sky is excluded, only the hood at the bottom survives). Detections that fall mostly inside
that region are the ego vehicle labeling itself and are dropped. The mask is cached in the object store as a
small downsampled grid keyed by vehicle+camera, computed once and reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core.logging import get_logger
from core.storage import get_object_store

log = get_logger("ego_mask")

GRID_H, GRID_W = 48, 64          # downsampled mask resolution stored per camera


@dataclass(frozen=True)
class EgoMask:
    grid: tuple[tuple[int, ...], ...]   # GRID_H x GRID_W of 0/1, 1 = ego hood
    area_frac: float

    def ego_fraction(self, bbox: tuple[float, float, float, float], frame_w: float,
                     frame_h: float) -> float:
        """How much of the box lies in the ego region, 0.0 to 1.0.

        The fraction rather than the verdict, because the two callers want different things from it.
        Deleting a box wants "is this the hood", which is the threshold below. Ranking a box for review
        wants the middle of the range: a box half on the bonnet is usually a reflection or a piece of the
        car read as an object, and it is exactly the one worth looking at, while a box fully on the hood
        has already been removed by the cleanup sweep and one fully off it is unremarkable.
        """
        if self.area_frac <= 0:
            return 0.0
        gh, gw = len(self.grid), len(self.grid[0])
        x1, y1, x2, y2 = bbox
        gx1 = max(0, min(gw - 1, int(x1 / frame_w * gw)))
        gx2 = max(0, min(gw, int(np.ceil(x2 / frame_w * gw))))
        gy1 = max(0, min(gh - 1, int(y1 / frame_h * gh)))
        gy2 = max(0, min(gh, int(np.ceil(y2 / frame_h * gh))))
        if gx2 <= gx1 or gy2 <= gy1:
            return 0.0
        cells = ego = 0
        for gy in range(gy1, gy2):
            for gx in range(gx1, gx2):
                cells += 1
                ego += self.grid[gy][gx]
        return (ego / cells) if cells else 0.0

    def contains_bbox(self, bbox: tuple[float, float, float, float], frame_w: float, frame_h: float,
                      frac: float = 0.5) -> bool:
        """True if at least `frac` of the box's area lies in the ego region (the box IS the hood)."""
        return self.ego_fraction(bbox, frame_w, frame_h) >= frac if self.area_frac > 0 else False


class WelfordStd:
    """Streaming mean and variance over frames, so the estimator never holds the stack.

    The array form materialises (T, H, W): at 1080p and 200 frames that is 1.7 GB of float32 before the
    std is taken, which is why the stack size was in practice capped low enough to make the estimate noisy.
    Welford's recurrence gets the same variance in one pass with two (H, W) accumulators, so the frame
    budget stops being a memory question.

    Numerically it is also the better answer: the sum-of-squares shortcut subtracts two large nearly-equal
    numbers and loses precision exactly where the variance is small, which is the hood.
    """

    def __init__(self, shape: tuple[int, int]):
        self.n = 0
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)

    def update(self, frame: np.ndarray) -> None:
        x = np.asarray(frame, dtype=np.float64)
        if x.shape != self._mean.shape:
            raise ValueError(f"frame is {x.shape}, expected {self._mean.shape}")
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (x - self._mean)

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self._mean)
        # Population std, matching np.std's default, so the streaming path and the array path agree.
        return np.sqrt(self._m2 / self.n)


def estimate_from_gray_frames(frames, *, var_thresh: float = 6.0, top_limit_frac: float = 0.45,
                              min_area_frac: float = 0.01) -> EgoMask | None:
    """The streaming form: an iterable of (H, W) grayscale frames rather than a stack.

    Identical result to estimate_from_gray_stack, which is what tests/test_ego_mask.py pins, and it never
    holds more than two (H, W) accumulators regardless of how many frames it is given.
    """
    acc: WelfordStd | None = None
    for f in frames:
        arr = np.asarray(f)
        if arr.ndim != 2:
            return None
        if acc is None:
            acc = WelfordStd(arr.shape)
        acc.update(arr)
    if acc is None or acc.n < 4:
        return None
    return _mask_from_std(acc.std, var_thresh=var_thresh, top_limit_frac=top_limit_frac,
                          min_area_frac=min_area_frac)


def estimate_from_gray_stack(stack: np.ndarray, *, var_thresh: float = 6.0, top_limit_frac: float = 0.45,
                             min_area_frac: float = 0.01) -> EgoMask | None:
    """Pure core: given a (T,H,W) stack of grayscale frames from one camera, return the hood mask or None.

    A pixel is hood if it is temporally static (low std) AND lies in the bottom band AND is connected to the
    bottom edge (flood from the bottom row). This rejects the equally-static sky, which is not bottom-anchored.

    Kept as the array entry point because four existing tests exercise it; both forms now share
    `_mask_from_std`, so the streaming path cannot drift from the one the tests pin.
    """
    if stack.ndim != 3 or stack.shape[0] < 4:
        return None
    std = stack.astype(np.float32).std(axis=0)
    return _mask_from_std(std, var_thresh=var_thresh, top_limit_frac=top_limit_frac,
                          min_area_frac=min_area_frac)


def _mask_from_std(std: np.ndarray, *, var_thresh: float, top_limit_frac: float,
                   min_area_frac: float) -> EgoMask | None:
    h, w = std.shape
    static = std < var_thresh
    top_cut = int((1.0 - top_limit_frac) * h)   # only the bottom top_limit_frac of rows may be hood
    static[:top_cut, :] = False
    if not static[h - 1].any():
        return None                              # nothing static along the very bottom -> no visible hood

    # Keep only the static region connected to the bottom edge.
    try:
        from scipy.ndimage import label

        lbl, n = label(static)
        bottom_labels = set(lbl[h - 1][static[h - 1]].tolist())
        bottom_labels.discard(0)
        if not bottom_labels:
            return None
        hood = np.isin(lbl, list(bottom_labels))
    except Exception:  # noqa: BLE001 - no SciPy: fall back to the raw bottom-band static mask
        hood = static

    if hood.mean() < min_area_frac:
        return None

    # Downsample to the stored grid: a cell is hood if any pixel in it is hood (max-pool).
    ys = np.linspace(0, h, GRID_H + 1, dtype=int)
    xs = np.linspace(0, w, GRID_W + 1, dtype=int)
    grid = tuple(
        tuple(int(hood[ys[gy]:ys[gy + 1], xs[gx]:xs[gx + 1]].any()) for gx in range(GRID_W))
        for gy in range(GRID_H)
    )
    area = float(np.mean([c for row in grid for c in row]))
    return EgoMask(grid=grid, area_frac=area)


def _key(vehicle_id: str, cam_id: str) -> str:
    return f"ego_masks/{vehicle_id}/{cam_id}.json"


def _serialize(m: EgoMask) -> bytes:
    return json.dumps({"grid": [list(r) for r in m.grid], "area_frac": m.area_frac,
                       "grid_h": GRID_H, "grid_w": GRID_W}).encode()


def _deserialize(data: bytes) -> EgoMask:
    d = json.loads(data)
    return EgoMask(grid=tuple(tuple(int(v) for v in row) for row in d["grid"]), area_frac=float(d["area_frac"]))


async def estimate_ego_mask(vehicle_id: str, cam_id: str, *, n: int = 80, force: bool = False) -> EgoMask | None:
    """Estimate (or reuse) the hood mask for one camera and cache it in the object store. Idempotent unless
    force=True. Samples up to n frames spread across the camera's capture."""
    from sqlalchemy import func, select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.recall.backends import load_image_bgr

    store = get_object_store()
    key = _key(vehicle_id, cam_id)
    if not force and store.exists(key):
        return _deserialize(store.get_bytes(key))

    maker = get_sessionmaker()
    async with maker() as db:
        # Sample a CONSECUTIVE run from this camera's busiest session, not frames spread across the whole
        # capture: over hours the hood's pixel values drift with lighting/reflections and it stops looking
        # static, but within one short run only the scene moves and the hood stands out cleanly.
        busiest = (await db.execute(
            select(Frame.session_id, func.count()).join(DbSession, DbSession.session_id == Frame.session_id)
            .where(DbSession.vehicle_id == vehicle_id, Frame.cam_id == cam_id)
            .group_by(Frame.session_id).order_by(func.count().desc()).limit(1))).first()
        if busiest is None:
            return None
        uris = (await db.execute(
            select(Frame.img_uri).where(Frame.session_id == busiest[0], Frame.cam_id == cam_id)
            .order_by(Frame.ts_ns).limit(n))).scalars().all()
    if len(uris) < 4:
        return None

    import cv2

    imgs = []
    for uri in uris:
        try:
            bgr = load_image_bgr(store, uri)
        except Exception:  # noqa: BLE001
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if imgs and gray.shape != imgs[0].shape:
            gray = cv2.resize(gray, (imgs[0].shape[1], imgs[0].shape[0]))
        imgs.append(gray)
    if len(imgs) < 4:
        return None

    mask = estimate_from_gray_stack(np.stack(imgs))
    if mask is None:
        log.info("ego_mask.none", vehicle=vehicle_id, cam=cam_id, frames=len(imgs))
        return None
    store.put_bytes(key, _serialize(mask), "application/json")
    log.info("ego_mask.estimated", vehicle=vehicle_id, cam=cam_id, frames=len(imgs), area_frac=round(mask.area_frac, 4))
    return mask


@lru_cache(maxsize=256)
def get_ego_mask(vehicle_id: str, cam_id: str) -> EgoMask | None:
    """Cached read of a previously-estimated hood mask, or None if none is cached for this camera."""
    store = get_object_store()
    key = _key(vehicle_id, cam_id)
    try:
        if store.exists(key):
            return _deserialize(store.get_bytes(key))
    except Exception:  # noqa: BLE001
        return None
    return None


def clear_cache() -> None:
    get_ego_mask.cache_clear()
