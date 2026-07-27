"""Per-slice evaluation computed from the prediction plane, not supplied by the caller.

The protected-slice gate (services/verdyx/verdict.py) decides whether a challenger may ship by comparing
per-slice metrics for slices like "pedestrian_night" and "autorickshaw_glare". Those numbers used to arrive
as a request-body dict on POST /verdyx/evaluate: nothing in the tree computed them. A safety gate whose
inputs are hand-typed is a gate in name only, since the number it reads has no causal link to the model it
is judging.

This module computes them. A slice is a named predicate over frame scene axes and object class, evaluated
against the same sealed gold set and the same immutable InferenceRun the aggregate metrics come from, so a
slice metric is reproducible and auditable in exactly the way docs/MEASUREMENT.md describes for the whole.
"""

from __future__ import annotations

import uuid as uuidlib
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.ap import average_precision
from core.accel.matching import match_detections
from core.logging import get_logger
from db.models import Frame, GoldSet, InferenceRun, Object, Prediction

log = get_logger("verdyx_slice_eval")

# The scene axes the frame model records. A slice predicate may constrain any of them.
SCENE_AXES = ("weather", "time_of_day", "road_type", "density")


@dataclass(frozen=True)
class SliceSpec:
    """A named subpopulation: frames matching every scene constraint, objects of any listed class.

    Both parts are optional, so a slice can be scene-only ("night"), class-only ("pedestrian"), or the
    conjunction the safety gate cares about ("pedestrian_night").
    """

    slice_id: str
    scene: dict[str, tuple[str, ...]] = field(default_factory=dict)   # axis -> accepted values
    class_names: tuple[str, ...] = ()

    def frame_matches(self, scene: dict | None) -> bool:
        if not self.scene:
            return True
        s = scene or {}
        return all(str(s.get(axis)) in accepted for axis, accepted in self.scene.items())


# The slices the AV pack's protected list names, defined once so the gate and the computation agree on what
# the words mean. "glare" is an adverse-lighting condition the scene model records under weather.
DEFAULT_SLICES: tuple[SliceSpec, ...] = (
    SliceSpec("pedestrian_night", scene={"time_of_day": ("night",)}, class_names=("pedestrian",)),
    SliceSpec("autorickshaw_glare", scene={"weather": ("glare",)}, class_names=("autorickshaw",)),
    SliceSpec("night", scene={"time_of_day": ("night",)}),
    SliceSpec("rain", scene={"weather": ("rain",)}),
    SliceSpec("fog", scene={"weather": ("fog",)}),
    SliceSpec("vru", class_names=("pedestrian", "rider", "cycle", "motorcycle")),
)


def _slices_for(names: list[str] | None) -> tuple[SliceSpec, ...]:
    if not names:
        return DEFAULT_SLICES
    by_id = {s.slice_id: s for s in DEFAULT_SLICES}
    out = []
    for n in names:
        if n in by_id:
            out.append(by_id[n])
        else:
            # An unknown protected slice must not silently evaluate to "fine". It is reported with
            # measured=False so the gate can refuse rather than pass on an absent number.
            out.append(SliceSpec(n))
    return tuple(out)


async def compute_slice_metrics(db: AsyncSession, gold_id: str, *, run_id: str,
                                slice_ids: list[str] | None = None,
                                iou_thr: float = 0.5, score_thr: float = 0.0) -> dict:
    """Compute {slice_id: {precision, recall, tp, fp, fn, support, measured}} for one run over one gold set.

    Ground truth is the sealed gold objects; predictions are the run's immutable Prediction rows. Matching is
    class-aware at `iou_thr`, the same rule the aggregate metrics use, so a slice number is directly
    comparable to the whole-set number rather than being a differently-defined statistic.
    """
    from services.autolabel.ontology import get_ontology

    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}
    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        return {"error": "gold set not found", "gold_id": gold_id}
    gold_ids = {UUID(str(o)) for o in (gold.object_ids or [])}
    if not gold_ids:
        return {"error": "gold set has no objects", "gold_id": gold_id}

    onto = get_ontology()
    specs = _slices_for(slice_ids)

    gold_rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
        .where(Object.object_id.in_(gold_ids)))).all()
    frame_ids = {fid for _, fid, _, _ in gold_rows}
    if not frame_ids:
        return {"error": "gold objects not found in the corpus", "gold_id": gold_id}

    scene_by_frame = dict((await db.execute(
        select(Frame.frame_id, Frame.scene).where(Frame.frame_id.in_(frame_ids)))).all())

    pred_rows = (await db.execute(
        select(Prediction.frame_id, Prediction.class_id, Prediction.bbox, Prediction.conf)
        .where(Prediction.run_id == run.run_id, Prediction.frame_id.in_(frame_ids)))).all()

    by_frame_gt: dict[uuidlib.UUID, list] = {}
    for _oid, fid, cid, bbox in gold_rows:
        by_frame_gt.setdefault(fid, []).append((int(cid), list(bbox)))
    by_frame_pred: dict[uuidlib.UUID, list] = {}
    for fid, cid, bbox, conf in pred_rows:
        if conf is not None and conf < score_thr:
            continue
        # a reconstructed run carries no confidence; order is then arbitrary but matching is still class-aware
        by_frame_pred.setdefault(fid, []).append((int(cid), list(bbox), float(conf) if conf is not None else 0.0))

    out: dict[str, dict] = {}
    for spec in specs:
        wanted = {onto.by_name(n).id for n in spec.class_names if onto.has_name(n)}
        if spec.class_names and not wanted:
            out[spec.slice_id] = {"measured": False, "reason": "no such class in the ontology"}
            continue

        tp = fp = fn = 0
        support = 0
        # Accumulate per-prediction (score, is_tp) across the slice's frames so AP is a real PR-curve
        # integral over the whole slice, not an average of per-frame numbers.
        scores: list[float] = []
        tp_flags: list[bool] = []
        for fid, gts in by_frame_gt.items():
            if not spec.frame_matches(scene_by_frame.get(fid)):
                continue
            g = [(c, b) for c, b in gts if not wanted or c in wanted]
            p = [(c, b, sc) for c, b, sc in by_frame_pred.get(fid, []) if not wanted or c in wanted]
            support += len(g)
            if not g and not p:
                continue
            m = match_detections([b for _, b, _ in p], [sc for _, _, sc in p], [c for c, _, _ in p],
                                 [b for _, b in g], [c for c, _ in g], iou_thr)
            tp += int(m["n_tp"])
            fp += int(m["n_fp"])
            fn += int(m["n_fn"])
            order = sorted(range(len(p)), key=lambda i: -p[i][2])
            for rank, i in enumerate(order):
                scores.append(p[i][2])
                tp_flags.append(bool(m["tp"][rank]))

        if support == 0:
            # No gold evidence for this slice. Reporting 0.0 would look like a failure and 1.0 like a pass;
            # both are lies. measured=False lets the gate treat an unevidenced protected slice as a refusal.
            out[spec.slice_id] = {"measured": False, "reason": "no gold objects in this slice",
                                  "support": 0}
            continue
        ap = average_precision(scores, tp_flags, support) if scores else 0.0
        out[spec.slice_id] = {
            "measured": True,
            # "map" is the key services/verdyx/verdict.py compares protected slices on; ap50 is the same
            # number under its precise name.
            "map": round(ap or 0.0, 4),
            "ap50": round(ap or 0.0, 4),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "support": support,
        }

    log.info("verdyx.slice_metrics", run_id=run_id, gold_id=gold_id,
             slices=len(out), measured=sum(1 for v in out.values() if v.get("measured")))
    return out
