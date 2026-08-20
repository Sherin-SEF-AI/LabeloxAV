"""Non-maximum suppression that knows two overlapping boxes can be two objects.

The live suppression (services/autolabel/fusion.py::_suppress_duplicates) decides that two boxes are the
same physical object when their classes are equal, share an l1 superclass, or one of them is a fallback
catch-all. The l1 clause is the problem, and on Indian roads it is an expensive one: a pedestrian and a
rider are both l1 "vru", so a pedestrian standing in front of a motorcyclist at IoU 0.6 is silently merged
into one box and one of them stops existing. The same clause merges a cyclist and the pedestrian they are
passing, and a cow and the person leading it.

That rule is a proxy for a question it cannot answer. "Could these be the same object" depends on the
class pair, not on whether the two classes happen to sit under the same superclass. Some pairs really are
redundant detections of one thing (sedan and vehicle_fallback), some pairs genuinely co-occur at high
overlap (rider and motorcycle, pedestrian and umbrella), and some are simply never both true of one box.

So the decision moves to a learned compatibility matrix: for an ordered class pair, the probability that
two boxes at this overlap are DISTINCT objects rather than one object seen twice. Suppression happens when
that probability is low. The matrix is estimated from human-confirmed co-occurrence
(services/autolabel/compat_matrix.py) with a Laplace prior, so a pair nobody has ever labelled together
falls back to the prior rather than to a confident guess.

TWO OUTPUTS, NOT ONE. A pair that survives suppression because it genuinely co-occurs is exactly a pair
that may stand in a relationship: a rider on a motorcycle, a person pulling a cart. Those are emitted as
provisional edges rather than discarded, because the overlap geometry that proved they were two objects is
the same evidence a scene-graph proposer would use, and computing it twice invites the two to disagree.

THE PRIOR IS VISIBLE. Every decision carries the support it rested on. On a corpus with 621 human objects
the matrix is almost entirely prior, and a caller that cannot see that would read "learned compatibility"
as a measurement.

NumPy is the oracle; the torch path mirrors the pairwise stage and must agree to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from core.accel.boxes import box_iou_matrix

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass(frozen=True)
class RelNmsResult:
    """What survived, what was dropped and why, and which surviving pairs may be related.

    `keep` indexes the input. `suppressed` pairs each dropped index with the index that absorbed it and the
    distinctness probability that allowed it. `edges` are ordered (subject, object) index pairs that
    overlapped enough to be worth proposing a relationship for, with the overlap that produced them.

    `n_prior_only` counts decisions made entirely on the Laplace prior, because a suppression rule that
    never says how much evidence it had is not distinguishable from a hard-coded one.
    """

    keep: npt.NDArray[np.int64]
    suppressed: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    n_prior_only: int


def _overlap_np(boxes: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """IoU and intersection-over-min, pairwise. IoM catches a small box nested inside a large one, which
    IoU scores low precisely when the nesting is most complete."""
    iou = box_iou_matrix(boxes, boxes)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    min_area = np.minimum(area[:, None], area[None, :])
    iom = np.where(min_area > 0, inter / np.maximum(min_area, 1e-12), 0.0)
    return iou, iom


def _overlap_torch(boxes: npt.NDArray[np.float64], device: str
                   ) -> tuple[npt.NDArray[np.float64],
                              npt.NDArray[np.float64]]:  # pragma: no cover - GPU path
    b = torch.as_tensor(boxes, device=device, dtype=torch.float64)
    x1 = torch.maximum(b[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(b[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(b[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(b[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area = (b[:, 2] - b[:, 0]).clamp(min=0.0) * (b[:, 3] - b[:, 1]).clamp(min=0.0)
    union = area[:, None] + area[None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp(min=1e-12), torch.zeros_like(inter))
    min_area = torch.minimum(area[:, None], area[None, :])
    iom = torch.where(min_area > 0, inter / min_area.clamp(min=1e-12), torch.zeros_like(inter))
    out_iou: npt.NDArray[np.float64] = iou.cpu().numpy()
    out_iom: npt.NDArray[np.float64] = iom.cpu().numpy()
    return out_iou, out_iom


def relationship_nms(boxes: npt.ArrayLike, scores: npt.ArrayLike, class_ids: npt.ArrayLike, *,
                     distinct_prob: Any, iou_thr: float = 0.7, iom_thr: float = 0.8,
                     distinct_floor: float = 0.5, edge_iou: float = 0.3,
                     relation_for_pair: Any = None, device: str | None = None) -> RelNmsResult:
    """Greedy NMS in descending score, suppressing only pairs the matrix says are one object.

    `distinct_prob(a_class, b_class)` returns (probability the two are distinct objects, support count).
    Suppression requires geometric overlap AND a distinctness probability below `distinct_floor`: both
    conditions, because overlap alone is what the current rule uses and it is the half that is wrong.

    `relation_for_pair(a_class, b_class)` optionally names the relationship an overlapping pair may stand
    in ("rider_of"). Pairs that overlap by at least `edge_iou` and survive suppression are emitted as
    provisional edges; nothing here writes them, and nothing here decides they are true.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    n = b.shape[0]
    if not (n == s.size == c.size):
        raise ValueError(f"{n} boxes, {s.size} scores, {c.size} classes")
    if n == 0:
        return RelNmsResult(keep=np.zeros(0, dtype=np.int64), suppressed=(), edges=(), n_prior_only=0)

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        iou, iom = _overlap_torch(b, str(device))  # pragma: no cover - GPU path
    else:
        iou, iom = _overlap_np(b)

    order = np.argsort(-s, kind="stable")
    keep: list[int] = []
    suppressed: list[dict[str, Any]] = []
    prior_only = 0

    for i in (int(x) for x in order):
        absorbed_by = None
        for k in keep:
            if not (iou[i, k] >= iou_thr or iom[i, k] >= iom_thr):
                continue
            p, support = distinct_prob(int(c[i]), int(c[k]))
            if support == 0:
                prior_only += 1
            if p < distinct_floor:
                absorbed_by = (k, float(p), int(support))
                break
        if absorbed_by is None:
            keep.append(i)
        else:
            k, p, support = absorbed_by
            suppressed.append({"index": i, "absorbed_by": k, "distinct_prob": round(p, 6),
                               "support": support, "iou": round(float(iou[i, k]), 4),
                               "iom": round(float(iom[i, k]), 4)})

    edges: list[dict[str, Any]] = []
    if relation_for_pair is not None:
        for ai, a in enumerate(keep):
            for bi, bkeep in enumerate(keep):
                if ai == bi or iou[a, bkeep] < edge_iou:
                    continue
                kind = relation_for_pair(int(c[a]), int(c[bkeep]))
                if kind:
                    edges.append({"from_index": a, "to_index": bkeep, "kind": kind,
                                  "iou": round(float(iou[a, bkeep]), 4),
                                  "iom": round(float(iom[a, bkeep]), 4)})

    return RelNmsResult(keep=np.asarray(sorted(keep), dtype=np.int64),
                        suppressed=tuple(suppressed), edges=tuple(edges), n_prior_only=prior_only)


__all__ = ["RelNmsResult", "relationship_nms"]
