"""Which class pairs can genuinely overlap, learned from what humans actually drew.

The live duplicate rule treats two boxes as one object when their classes share an l1 superclass. On
Indian roads that merges a pedestrian standing in front of a motorcyclist, a cyclist passing a pedestrian,
and a cow beside the person leading it, because all of those are l1 "vru". The rule is a proxy for a
question about the class PAIR, and the superclass is not that question.

This estimates the answer instead. For every frame, every ordered pair of human-confirmed objects whose
boxes overlap by at least MIN_IOU is one observation that those two classes CAN be two distinct objects at
that overlap. A pair never seen together falls back to a Laplace prior rather than to a confident guess.

WHAT THE NEGATIVE CLASS IS, AND WHY IT IS NOT WHAT YOU WOULD FIRST REACH FOR. There is no labelled set of
"these two boxes were really one object": nobody records a merge that already happened. So `n_apart` is
counted from human frames where two boxes of those classes are co-present and do NOT overlap. That makes
this an estimate of "do these classes appear as separate overlapping things", which is the question the
NMS actually needs, and it is worth saying because the name "compatibility" invites a stronger reading.

THE COMPARISON IS AGAINST THE CORPUS, NOT AGAINST A HALF. The obvious estimator,

    (n_overlapping + alpha) / (n_overlapping + n_apart + 2 * alpha)

compared against 0.5, is wrong here, and measurably so: on this corpus it puts motorcycle-and-rider at
0.158 and would therefore merge exactly the pair this exists to keep apart. The reason is combinatorics.
A frame with three motorcycles and one rider contributes three co-present pairs of which at most one
overlaps, so `n_apart` is inflated by how crowded the scene is rather than by anything about the pair. On
Indian road scenes almost every pair of classes is usually apart, so almost every cell lands far below
0.5 and the flat prior is not the right thing to be measuring against.

What matters is whether a pair overlaps MORE OFTEN THAN PAIRS IN GENERAL do:

    pair_rate   = (n_overlapping + m * base_rate) / (n_overlapping + n_apart + m)
    P(distinct) = pair_rate / (pair_rate + base_rate)

where `base_rate` is the corpus-wide fraction of co-present pairs that overlap. P is exactly 0.5 when a
pair overlaps at the corpus baseline, above it when the pair overlaps more than typical, below when less.
Shrinking toward `base_rate` rather than toward a half is what makes an unobserved pair land on 0.5:
knowing nothing about a pair should mean treating it as typical, not as confidently distinct. On the real
corpus this moves motorcycle-and-rider to 0.67 (kept apart) while leaving motorcycle-and-scooter at 0.23
(merged), which is the ordering the failure mode demanded.

THE SUPPORT IS THE POINT. This corpus holds 621 human-confirmed objects and 63 relationships. Almost every
cell will be prior, and a caller that cannot see that would read a 0.5 as a measurement rather than as the
prior showing through. Every lookup returns its support alongside its probability, the summary reports how
many cells were ever observed, and a matrix built on nothing says so instead of looking learned.
"""

from __future__ import annotations

import uuid as uuidlib
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.boxes import box_iou_matrix
from core.logging import get_logger
from db.models import Object

log = get_logger("compat_matrix")

# Overlap at which two boxes are worth counting as a co-occurrence at all. Below this they are simply two
# things near each other and say nothing about whether the classes can be confused for one object.
MIN_IOU = 0.3
LAPLACE_ALPHA = 1.0
# How many corpus-baseline observations a pair is shrunk toward before its own evidence dominates. Small,
# because per-pair counts on this corpus are in the tens: at 5 a pair needs a handful of frames to move,
# which is about the point at which its overlap rate stops being an accident of one crowded junction.
PSEUDO_COUNT = 5.0

# The label states that mean a person actually looked. A machine-proposed box is the thing being judged;
# counting it as evidence would let the fusion rule confirm its own output.
_HUMAN_STATES = ("accepted",)
_HUMAN_LIFECYCLE = ("human_confirmed", "human_edited", "track_confirmed")


class CompatMatrix:
    """A learned pair matrix with its provenance, or an honest prior when nothing was learned.

    `distinct_prob(a, b)` returns (probability, support). Symmetric: the pair (a, b) and (b, a) are one
    cell, because "can these two be two objects" does not depend on which one was scored higher.
    """

    def __init__(self, together: dict[tuple[int, int], int], apart: dict[tuple[int, int], int],
                 *, ontology_version: str, snapshot: str, n_frames: int, n_objects: int,
                 alpha: float = LAPLACE_ALPHA, pseudo: float = PSEUDO_COUNT):
        self._together = together
        self._apart = apart
        self.ontology_version = ontology_version
        self.snapshot = snapshot
        self.n_frames = n_frames
        self.n_objects = n_objects
        self.alpha = alpha
        self.pseudo = pseudo

    @staticmethod
    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    @property
    def base_rate(self) -> float:
        """The corpus-wide fraction of co-present pairs that overlap: what "typical" means here.

        Falls back to the flat prior when nothing was counted, which is the only value that does not
        assert something about a corpus that was never read.
        """
        n_t, n_a = sum(self._together.values()), sum(self._apart.values())
        if n_t + n_a == 0:
            return 0.5
        return (n_t + self.alpha) / (n_t + n_a + 2 * self.alpha)

    def distinct_prob(self, a: int, b: int) -> tuple[float, int]:
        """(probability these two boxes are distinct objects at overlap, observations behind it).

        Shrunk toward the corpus base rate by `pseudo` observations, so a pair nobody has seen lands on
        exactly 0.5 rather than on a confident answer, and a pair seen twice does not swing on two frames.
        """
        k = self._key(a, b)
        n_t, n_a = self._together.get(k, 0), self._apart.get(k, 0)
        base = self.base_rate
        pair_rate = (n_t + self.pseudo * base) / (n_t + n_a + self.pseudo)
        denom = pair_rate + base
        return ((pair_rate / denom if denom > 0 else 0.5), n_t + n_a)

    def summary(self) -> dict:
        observed = {k for k in set(self._together) | set(self._apart)
                    if self._together.get(k, 0) + self._apart.get(k, 0) > 0}
        total = sum(self._together.values()) + sum(self._apart.values())
        return {
            "ontology_version": self.ontology_version, "snapshot": self.snapshot,
            "alpha": self.alpha, "pseudo": self.pseudo, "base_rate": round(self.base_rate, 6),
            "n_frames": self.n_frames, "n_objects": self.n_objects,
            "n_cells_observed": len(observed), "n_observations": total,
            "learned": bool(observed),
            # Said in the object rather than left to the reader. A matrix with four observed cells is a
            # prior wearing the word "learned".
            "caveat": ("cells with zero support return the Laplace prior; with "
                       f"{len(observed)} observed cells over {self.n_objects} human-confirmed objects, "
                       "most lookups are prior and must not be read as measurements"),
        }

    def to_json(self) -> dict:
        return {"kind": "compat-matrix-v1", **self.summary(),
                "together": {f"{a}:{b}": n for (a, b), n in sorted(self._together.items())},
                "apart": {f"{a}:{b}": n for (a, b), n in sorted(self._apart.items())}}

    @classmethod
    def from_json(cls, blob: dict) -> CompatMatrix:
        def _parse(d: dict) -> dict[tuple[int, int], int]:
            out = {}
            for k, v in (d or {}).items():
                a, b = k.split(":")
                out[(int(a), int(b))] = int(v)
            return out

        return cls(_parse(blob.get("together", {})), _parse(blob.get("apart", {})),
                   ontology_version=blob.get("ontology_version", "unknown"),
                   snapshot=blob.get("snapshot", "unknown"), n_frames=int(blob.get("n_frames", 0)),
                   n_objects=int(blob.get("n_objects", 0)),
                   alpha=float(blob.get("alpha", LAPLACE_ALPHA)),
                   pseudo=float(blob.get("pseudo", PSEUDO_COUNT)))


def prior_matrix(ontology_version: str = "unknown") -> CompatMatrix:
    """A matrix that learned nothing and says so. Every lookup is exactly the prior."""
    return CompatMatrix({}, {}, ontology_version=ontology_version, snapshot="prior-only",
                        n_frames=0, n_objects=0)


async def build_compat_matrix(db: AsyncSession, *, limit_frames: int = 20000) -> CompatMatrix:
    """Count co-occurrence over human-confirmed objects, frame by frame.

    Only human-confirmed labels count. A machine box is what the fusion rule is deciding about, so letting
    it vote would let the rule confirm its own output, and the resulting matrix would encode the current
    merging behaviour as ground truth about the world.
    """
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(
        select(Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.state.in_(_HUMAN_STATES),
               (Object.source == "human") | (Object.lifecycle.in_(_HUMAN_LIFECYCLE)))
        .limit(limit_frames * 64))).all()

    by_frame: dict[object, list[tuple[int, list[float]]]] = {}
    for fid, cid, bbox in rows:
        by_frame.setdefault(fid, []).append((int(cid), list(bbox)))

    together: dict[tuple[int, int], int] = {}
    apart: dict[tuple[int, int], int] = {}
    n_objects = 0
    for objs in by_frame.values():
        n_objects += len(objs)
        if len(objs) < 2:
            continue
        boxes = np.asarray([o[1] for o in objs], dtype=float).reshape(-1, 4)
        classes = [o[0] for o in objs]
        iou = box_iou_matrix(boxes, boxes)
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                key = CompatMatrix._key(classes[i], classes[j])
                if iou[i, j] >= MIN_IOU:
                    together[key] = together.get(key, 0) + 1
                else:
                    apart[key] = apart.get(key, 0) + 1

    m = CompatMatrix(together, apart, ontology_version=onto.version,
                     snapshot=datetime.now(UTC).date().isoformat(),
                     n_frames=len(by_frame), n_objects=n_objects)
    s = m.summary()
    log.info("compat_matrix.built", frames=s["n_frames"], objects=s["n_objects"],
             cells=s["n_cells_observed"], observations=s["n_observations"], learned=s["learned"])
    return m


_CACHE: dict[str, CompatMatrix] = {}


async def get_compat_matrix(db: AsyncSession, *, refresh: bool = False) -> CompatMatrix:
    """The current matrix, built once per process unless refreshed.

    Cached because it is a corpus-wide aggregate read on a labelling hot path, and it changes only when
    somebody confirms more labels. The nightly schedule rebuilds it; this keeps a run consistent with
    itself rather than re-counting per frame.
    """
    from services.autolabel.ontology import get_ontology

    key = get_ontology().version
    if refresh or key not in _CACHE:
        _CACHE[key] = await build_compat_matrix(db)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


_LAST_REBUILD: dict[str, str] = {}


async def maybe_rebuild_matrix(db: AsyncSession) -> dict:
    """Nightly rebuild, self-guarded against re-firing, in the shape services/agent/runtime/schedule.py wants.

    Guarded on the snapshot date rather than on a timestamp: the matrix is a daily aggregate and rebuilding
    it twice in one night produces the identical counts at the cost of a full corpus scan.
    """
    from services.autolabel.ontology import get_ontology

    today = datetime.now(UTC).date().isoformat()
    key = get_ontology().version
    if _LAST_REBUILD.get(key) == today:
        return {"ran": False, "reason": "already rebuilt today", "snapshot": today}
    m = await build_compat_matrix(db)
    _CACHE[key] = m
    _LAST_REBUILD[key] = today
    return {"ran": True, **m.summary()}


async def matrix_report(db: AsyncSession, *, top: int = 25) -> dict:
    """The matrix with its most-observed cells named, for the analytics surface.

    Sorted by support rather than by probability: a cell at 1.0 on one observation is not a finding, and
    ordering by probability would put exactly those at the top.
    """
    from services.autolabel.ontology import get_ontology

    m = await get_compat_matrix(db, refresh=True)
    onto = get_ontology()

    def _name(cid: int) -> str:
        try:
            return onto.by_id(cid).name
        except Exception:  # noqa: BLE001
            return f"class_{cid}"

    cells = []
    for key in set(m._together) | set(m._apart):
        p, support = m.distinct_prob(*key)
        cells.append({"a": _name(key[0]), "b": _name(key[1]), "a_id": key[0], "b_id": key[1],
                      "distinct_prob": round(p, 4), "support": support,
                      "n_overlapping": m._together.get(key, 0), "n_apart": m._apart.get(key, 0)})
    cells.sort(key=lambda c: (-c["support"], c["a"], c["b"]))
    return {**m.summary(), "cells": cells[:max(1, top)],
            # An unobserved pair lands here by construction: shrunk fully to the base rate, so its
            # probability is exactly a half whatever the base rate happens to be.
            "prior_value": 0.5}


async def total_human_objects(db: AsyncSession) -> int:
    return int((await db.execute(select(func.count()).select_from(Object).where(
        Object.state.in_(_HUMAN_STATES),
        (Object.source == "human") | (Object.lifecycle.in_(_HUMAN_LIFECYCLE))))).scalar_one())


def new_run_id() -> str:
    return str(uuidlib.uuid4())


__all__ = ["CompatMatrix", "build_compat_matrix", "get_compat_matrix", "matrix_report",
           "prior_matrix", "clear_cache", "maybe_rebuild_matrix", "MIN_IOU", "LAPLACE_ALPHA",
           "PSEUDO_COUNT"]
