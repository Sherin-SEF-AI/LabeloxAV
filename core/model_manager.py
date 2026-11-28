"""One place that loads, shares and frees inference models, and one way to ask how much VRAM is left.

Models were constructed wherever they were needed: `YOLO(...)` at fourteen inference sites, SAM four
separate ways, each behind its own module global, and four different readings of free VRAM. On one 16 GB
card that had three costs. The same weights were resident more than once in a process (the editor's SAM,
the describe path's SAM and Path B's SAM are one file). Nothing could free a model somebody else had
loaded, so a job that needed the room failed instead of making it. And nobody could say what was on the
card.

This is the registry those sites share.

    model = manager().get("sam:sam2_b.pt", lambda: SAM("sam2_b.pt"), est_mb=1200)

`get` loads on first use and returns the shared instance afterwards. `lease` does the same and holds the
model for the length of a block, so nothing evicts it mid-batch. `release` frees one by key; `make_room`
evicts idle models, least recently used first, until a requested amount of VRAM is free. `stats` says what
is loaded.

What does not belong here: a model about to be trained or exported. Training mutates the weights it is
handed, so those call sites keep constructing their own, private instance.

Thread-safe. Loading takes seconds and runs outside the registry lock, under a lock of its own per key, so
two threads asking for the same model load it once and a thread asking for a different model is not held
up behind them.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from core.logging import get_logger

log = get_logger("model_manager")


class GpuCapacityError(RuntimeError):
    """A load was refused because it would not fit in the VRAM that is free."""


@dataclass
class _Entry:
    key: str
    model: Any
    est_mb: float
    device: str | None
    loaded_at: float
    last_used: float
    leases: int = 0
    pinned: bool = False
    load_s: float = 0.0
    uses: int = 0
    on_evict: Callable[[Any], None] | None = field(default=None, repr=False)


def device_index(device: str | int | None) -> int:
    """The CUDA ordinal in a device string: "cuda:1" -> 1, "cuda" / None -> 0."""
    if isinstance(device, int):
        return device
    if device and ":" in str(device):
        try:
            return int(str(device).split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch in this image is simply "no GPU here"
        return False


def vram_free_mb(device: str | int | None = None) -> float | None:
    """Free VRAM on one device as the driver reports it, in MB. None when there is no CUDA device.

    The single probe. There were four: one read `mem_get_info()` with no device index and so always
    answered for device 0, which is the wrong card as soon as there is a second.
    """
    if not cuda_available():
        return None
    import torch

    idx = device_index(device)
    try:
        free, _total = torch.cuda.mem_get_info(idx)
    except Exception:  # noqa: BLE001 - an invalid ordinal or an uninitialised context
        try:
            init: Any = torch.cuda.init      # typed in some torch builds and not others; CI has the other
            init()
            free, _total = torch.cuda.mem_get_info(idx)
        except Exception:  # noqa: BLE001
            return None
    return float(free) / (1024 * 1024)


def vram_total_mb(device: str | int | None = None) -> float | None:
    if not cuda_available():
        return None
    import torch

    try:
        _free, total = torch.cuda.mem_get_info(device_index(device))
    except Exception:  # noqa: BLE001
        return None
    return float(total) / (1024 * 1024)


@contextlib.contextmanager
def keep_gpu_visible() -> Iterator[None]:
    """Undo what an ultralytics predict does to the process environment.

    Ultralytics' `select_device` writes CUDA_VISIBLE_DEVICES for the whole process on every predict: "-1"
    for device="cpu", the index otherwise. The privacy plate detector runs on the CPU, so after the first
    ingested frame, any process that had not yet touched CUDA could no longer see the GPU, and auto-label
    in that process refused with "CUDA not available". Wrap every ultralytics predict that may run on the
    CPU in this.

    Restoring the variable afterwards is not enough on its own: the CUDA runtime reads it once, when the
    process first touches CUDA, and torch caches the answer. If that first touch happens inside the CPU
    predict, the GPU stays hidden for the life of the process. So CUDA is initialised before the predict.
    """
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    if before != "-1":
        cuda_available()                         # first touch happens here, with the GPU still visible
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = before


def empty_cuda_cache() -> None:
    """Return freed blocks to the driver, so the next probe (ours or another process's) sees them."""
    if not cuda_available():
        return
    import gc

    import torch

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


class ModelManager:
    def __init__(self, *, free_mb: Callable[[str | int | None], float | None] = vram_free_mb,
                 empty_cache: Callable[[], None] = empty_cuda_cache,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._load_locks: dict[str, threading.Lock] = {}
        self._infer_locks: dict[str, threading.Lock] = {}
        self._free_mb = free_mb
        self._empty_cache = empty_cache
        self._clock = clock

    # ---- loading -------------------------------------------------------------------------------------

    def get(self, key: str, loader: Callable[[], Any], *, est_mb: float = 0.0, device: str | None = None,
            pinned: bool = False, headroom_mb: float = 0.0,
            on_evict: Callable[[Any], None] | None = None) -> Any:
        """The shared model for `key`, loading it with `loader` on first use.

        `est_mb` is what the model is expected to take. When it is given and a GPU is present, the load
        is refused (after evicting idle models to try to make room) if it would leave less than
        `headroom_mb` free. `pinned` models are never evicted by `make_room`.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used = self._clock()
                entry.uses += 1
                return entry.model
            load_lock = self._load_locks.setdefault(key, threading.Lock())

        with load_lock:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:            # another thread loaded it while this one waited
                    entry.last_used = self._clock()
                    entry.uses += 1
                    return entry.model
            if est_mb > 0:
                self.require(est_mb, key, device=device, headroom_mb=headroom_mb)
            t0 = self._clock()
            model = loader()
            now = self._clock()
            with self._lock:
                self._entries[key] = _Entry(key=key, model=model, est_mb=float(est_mb), device=device,
                                            loaded_at=now, last_used=now, pinned=pinned,
                                            load_s=now - t0, uses=1, on_evict=on_evict)
            log.info("model.loaded", key=key, est_mb=round(est_mb), load_s=round(now - t0, 2),
                     device=device)
            return model

    @contextlib.contextmanager
    def lease(self, key: str, loader: Callable[[], Any], **kwargs: Any) -> Iterator[Any]:
        """`get`, and hold the model for the block: a leased model is never evicted."""
        model = self.get(key, loader, **kwargs)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases += 1
        try:
            yield model
        finally:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.leases = max(0, entry.leases - 1)
                    entry.last_used = self._clock()

    def infer_lock(self, key: str) -> threading.Lock:
        """The lock to hold around a predict call on a shared model.

        Sharing an instance means two callers can reach it at once, and an Ultralytics predictor keeps
        per-call state on the object (the image it has embedded, the prompts it was given). A click in
        the editor while an auto-label batch is mid-frame would otherwise read the other's image. One
        lock per key, so different models still run side by side.
        """
        with self._lock:
            return self._infer_locks.setdefault(key, threading.Lock())

    # ---- freeing -------------------------------------------------------------------------------------

    def release(self, key: str, *, force: bool = False) -> bool:
        """Drop one model. Refused while it is leased unless `force`. True when something was dropped."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.leases > 0 and not force:
                return False
            del self._entries[key]
        self._dispose(entry)
        self._empty_cache()
        log.info("model.released", key=key, est_mb=round(entry.est_mb))
        return True

    def release_prefix(self, prefix: str, *, force: bool = False) -> int:
        with self._lock:
            keys = [k for k in self._entries if k.startswith(prefix)]
        return sum(1 for k in keys if self.release(k, force=force))

    def release_all(self, *, force: bool = False) -> int:
        return self.release_prefix("", force=force)

    def _dispose(self, entry: _Entry) -> None:
        if entry.on_evict is not None:
            try:
                entry.on_evict(entry.model)
            except Exception as exc:  # noqa: BLE001 - a failing hook must not leave the model resident
                log.warning("model.on_evict_failed", key=entry.key, error=str(exc))
        entry.model = None

    def make_room(self, need_mb: float, *, device: str | None = None, headroom_mb: float = 0.0) -> bool:
        """Evict idle, unpinned models, least recently used first, until `need_mb` + headroom is free.

        True when enough is free afterwards (or there is no GPU to measure). Models on a different device
        are left alone: evicting them frees nothing on the card that is short.
        """
        free = self._free_mb(device)
        if free is None:
            return True
        while free - need_mb < headroom_mb:
            with self._lock:
                idle = sorted((e for e in self._entries.values()
                               if e.leases == 0 and not e.pinned
                               and device_index(e.device) == device_index(device)),
                              key=lambda e: e.last_used)
                victim = idle[0] if idle else None
                if victim is not None:
                    del self._entries[victim.key]
            if victim is None:
                return False
            self._dispose(victim)
            self._empty_cache()
            log.info("model.evicted", key=victim.key, est_mb=round(victim.est_mb), to_fit_mb=round(need_mb))
            free = self._free_mb(device)
            if free is None:
                return True
        return True

    def require(self, need_mb: float, name: str, *, device: str | None = None,
                headroom_mb: float = 0.0) -> None:
        """Raise GpuCapacityError unless `need_mb` fits, after evicting whatever idle models it takes."""
        free = self._free_mb(device)
        if free is None:
            return
        if free - need_mb >= headroom_mb:
            return
        if self.make_room(need_mb, device=device, headroom_mb=headroom_mb):
            return
        free = self._free_mb(device) or 0.0
        raise GpuCapacityError(
            f"refusing to load {name}: need ~{need_mb:.0f} MB, free {free:.0f} MB, headroom "
            f"{headroom_mb:.0f} MB, and nothing idle is left to evict")

    # ---- inspection ----------------------------------------------------------------------------------

    def loaded(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def stats(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            return [{"key": e.key, "est_mb": round(e.est_mb), "device": e.device, "leases": e.leases,
                     "pinned": e.pinned, "uses": e.uses, "load_s": round(e.load_s, 2),
                     "idle_s": round(now - e.last_used, 1), "age_s": round(now - e.loaded_at, 1)}
                    for e in sorted(self._entries.values(), key=lambda e: e.key)]


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def manager() -> ModelManager:
    """The process-wide registry."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelManager()
    return _manager
