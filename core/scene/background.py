"""Per-camera background prior for a fixed (static) camera.

A moving camera has no stable background: every pixel changes as the vehicle drives. A fixed CCTV camera does,
and that background is the substrate for the static-camera scene model: foreground (people, vehicles) is what
differs from it, so a background prior turns "what moved" into a cheap, model-free signal for curation and
motion gating.

Two estimators, CPU reference implementations:

* temporal_median - the per-pixel median over a sample of frames. Robust to transient foreground (anything
  that is not present in most frames averages out), deterministic, no state. The default.
* Mog2Background - a thin wrapper over OpenCV's MOG2 mixture-of-Gaussians subtractor for the streaming case
  (adapts to slow lighting change). Optional; falls back to nothing if the cv2 build lacks it.

Both are pack-agnostic: they know nothing about ontologies or roads.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def temporal_median(frames: Sequence[np.ndarray], max_samples: int = 64) -> np.ndarray:
    """Per-pixel median background over up to max_samples evenly-spaced frames.

    Foreground that occupies any given pixel in a minority of frames is rejected by the median, so a person
    walking across a scene leaves no trace in the prior. Returns a background image of the same HxW(xC) shape
    and dtype as the inputs. Deterministic given the same frames.
    """
    seq = list(frames)
    if not seq:
        raise ValueError("temporal_median needs at least one frame")
    shape, dtype = seq[0].shape, seq[0].dtype
    for f in seq:
        if f.shape != shape:
            raise ValueError(f"all frames must share a shape; got {f.shape} vs {shape}")
    if len(seq) > max_samples:
        idx = np.linspace(0, len(seq) - 1, max_samples).round().astype(int)
        seq = [seq[i] for i in idx]
    stack = np.stack(seq, axis=0)
    med = np.median(stack, axis=0)
    # median of an even count is a .5 average; round back into the source dtype so the prior is a real image.
    if np.issubdtype(dtype, np.integer):
        return np.rint(med).astype(dtype)
    return med.astype(dtype)


def foreground_mask(frame: np.ndarray, background: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """Boolean HxW mask of pixels that differ from the background prior by more than `threshold` (per-pixel L1
    over channels). The model-free "what moved" signal a static camera gets for free."""
    if frame.shape != background.shape:
        raise ValueError(f"frame {frame.shape} and background {background.shape} must match")
    diff = np.abs(frame.astype(np.float32) - background.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    return diff > float(threshold)


class Mog2Background:
    """Streaming mixture-of-Gaussians background model (OpenCV MOG2). Adapts to slow lighting change, unlike
    the static median. Optional: raises at construction if this cv2 build has no MOG2, so a caller can fall
    back to temporal_median rather than get a silently degraded prior."""

    def __init__(self, history: int = 200, var_threshold: float = 16.0, detect_shadows: bool = False) -> None:
        import cv2

        factory = getattr(cv2, "createBackgroundSubtractorMOG2", None)
        if factory is None:  # pragma: no cover - depends on the cv2 build
            raise RuntimeError("this OpenCV build has no MOG2; use temporal_median instead")
        self._sub = factory(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Feed one frame; return its foreground mask (uint8 0/255)."""
        return self._sub.apply(frame)

    def fit(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Run a sequence through the model and return the learned background image."""
        for f in frames:
            self._sub.apply(f)
        return self._sub.getBackgroundImage()
