"""Specialists that learn from human rulings: corpus memory (nearest neighbours) and a linear probe.

Both read the SigLIP 2 / DINOv3 vectors already stored per object (366k of them), so neither runs a model at
decision time. Both are trained only on what people ruled, never on machine labels: a probe fitted to the
detector's output would learn the detector's mistakes and then vote for them with confidence.

**Memory** answers "what did people call things that look like this?" It is a query, not a model, so a
correction joins it the moment it is saved. Two guards keep it honest:
- when it is graded, neighbours from the same vehicle within `NEAR_NS` (15 minutes) of the object are
  excluded, because overlapping dashcam clips put the same hoarding in several sessions and a neighbour that
  is the object itself is not memory; at most `per_track` come from any one track, because 30 frames of one
  wrongly-labelled bus would otherwise outvote everything;
- it reports its margin, so a 50/50 neighbourhood reads as "unsure" rather than as a vote.

**The probe** is a multinomial logistic head over the concatenated vectors, with `not_object` as one of its
outputs. Classes with too few rulings are left out of it, and it abstains on them rather than guessing.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.labelagent.truth import TruthItem

log = get_logger("labelagent.memory")

NEAR_NS = 15 * 60 * 1_000_000_000


@dataclass
class Vectors:
    """Per-object vectors and the track each object belongs to, aligned by object id."""
    siglip: dict[str, np.ndarray] = field(default_factory=dict)
    dino: dict[str, np.ndarray] = field(default_factory=dict)
    track: dict[str, str | None] = field(default_factory=dict)


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


async def load_vectors(db: AsyncSession, object_ids: list[str]) -> Vectors:
    from db.models import Object, ObjectEmbedding

    out = Vectors()
    for k in range(0, len(object_ids), 2000):
        ids = [uuid.UUID(o) for o in object_ids[k:k + 2000]]
        for oid, sv, dv, tid in (await db.execute(
                select(ObjectEmbedding.object_id, ObjectEmbedding.siglip_vec, ObjectEmbedding.dino_vec,
                       Object.track_id)
                .join(Object, Object.object_id == ObjectEmbedding.object_id)
                .where(ObjectEmbedding.object_id.in_(ids)))).all():
            key = str(oid)
            if sv is not None:
                out.siglip[key] = _unit(sv)
            if dv is not None:
                out.dino[key] = _unit(dv)
            out.track[key] = str(tid) if tid else None
    return out


# ------------------------------------------------------------------------------------------------ memory

@dataclass
class Vote:
    label: int | None          # None when there were no usable neighbours
    share: float = 0.0         # weight of the winning label over all neighbour weight
    margin: float = 0.0        # winning share minus the runner-up's
    neighbours: list[dict] = field(default_factory=list)


@dataclass
class Bank:
    ids: list[str]
    labels: np.ndarray
    sessions: list[str]
    tracks: list[str | None]
    vecs: np.ndarray           # N x D, unit rows
    vehicles: list[str] = field(default_factory=list)
    ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def build(cls, items: list[TruthItem], vectors: Vectors) -> Bank:
        keep = [i for i in items if i.object_id in vectors.siglip]
        if not keep:
            return cls([], np.zeros(0, dtype=np.int64), [], [], np.zeros((0, 1), dtype=np.float32))
        return cls(ids=[i.object_id for i in keep],
                   labels=np.array([i.truth for i in keep], dtype=np.int64),
                   sessions=[i.session_id for i in keep],
                   tracks=[vectors.track.get(i.object_id) for i in keep],
                   vecs=np.stack([vectors.siglip[i.object_id] for i in keep]),
                   vehicles=[i.vehicle_id for i in keep],
                   ts=np.array([i.ts_ns for i in keep], dtype=np.int64))

    def vote(self, q: np.ndarray, *, k: int = 15, exclude_session: str | None = None,
             exclude_object: str | None = None, near: tuple[str, int] | None = None, near_ns: int = NEAR_NS,
             per_track: int = 2, temperature: float = 0.05) -> Vote:
        """`near=(vehicle_id, ts_ns)` drops neighbours from that vehicle within `near_ns` of that time."""
        if len(self.ids) == 0:
            return Vote(None)
        sims = self.vecs @ q
        order = np.argsort(-sims)
        picked: list[int] = []
        per: Counter = Counter()
        for j in order:
            if self.ids[j] == exclude_object or (exclude_session and self.sessions[j] == exclude_session):
                continue
            if near is not None and self.vehicles and self.vehicles[j] == near[0] \
                    and abs(int(self.ts[j]) - near[1]) < near_ns:
                continue
            t = self.tracks[j]
            if t is not None:
                if per[t] >= per_track:
                    continue
                per[t] += 1
            picked.append(int(j))
            if len(picked) >= k:
                break
        if not picked:
            return Vote(None)
        s = sims[picked]
        w = np.exp((s - s.max()) / temperature)
        tally: dict[int, float] = defaultdict(float)
        for j, wj in zip(picked, w, strict=True):
            tally[int(self.labels[j])] += float(wj)
        total = sum(tally.values())
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        top_share = ranked[0][1] / total
        second = ranked[1][1] / total if len(ranked) > 1 else 0.0
        return Vote(label=ranked[0][0], share=round(top_share, 4), margin=round(top_share - second, 4),
                    neighbours=[{"object_id": self.ids[j], "label": int(self.labels[j]),
                                 "sim": round(float(sims[j]), 4)} for j in picked])


# ------------------------------------------------------------------------------------------------ probe

def features(object_id: str, vectors: Vectors) -> np.ndarray | None:
    """The probe's input: SigLIP 2 and DINOv3 side by side. SigLIP alone when DINOv3 is missing would put
    a vector of a different length through the same head, so an object needs both or neither."""
    s, d = vectors.siglip.get(object_id), vectors.dino.get(object_id)
    if s is None or d is None:
        return None
    return np.concatenate([s, d])


@dataclass
class Probe:
    model: object
    classes: list[int]
    min_per_class: int

    @classmethod
    def fit(cls, items: list[TruthItem], vectors: Vectors, *, min_per_class: int = 8, c: float = 1.0) -> Probe:
        """Fit on human rulings. Track rulings repeat one object across many frames, so each item is
        weighted by 1/sqrt(its track's size) and one long track cannot dominate a class."""
        from sklearn.linear_model import LogisticRegression

        rows = [(i, f) for i in items if (f := features(i.object_id, vectors)) is not None]
        counts = Counter(i.truth for i, _ in rows)
        rows = [(i, f) for i, f in rows if counts[i.truth] >= min_per_class]
        if len({i.truth for i, _ in rows}) < 2:
            raise ValueError("too few ruled classes to fit a probe")
        track_n = Counter(vectors.track.get(i.object_id) for i, _ in rows)
        w = np.array([1.0 / np.sqrt(track_n[vectors.track.get(i.object_id)])
                      if vectors.track.get(i.object_id) else 1.0 for i, _ in rows])
        X = np.stack([f for _, f in rows])
        y = np.array([i.truth for i, _ in rows])
        m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        m.fit(X, y, sample_weight=w)
        return cls(model=m, classes=[int(x) for x in m.classes_], min_per_class=min_per_class)

    def predict(self, object_id: str, vectors: Vectors) -> tuple[int | None, float, float]:
        """(label, probability, margin over the runner-up), or (None, 0, 0) without vectors."""
        f = features(object_id, vectors)
        if f is None:
            return None, 0.0, 0.0
        p = self.model.predict_proba(f[None, :])[0]
        order = np.argsort(-p)
        top, second = float(p[order[0]]), float(p[order[1]]) if len(order) > 1 else 0.0
        return self.classes[int(order[0])], round(top, 4), round(top - second, 4)
