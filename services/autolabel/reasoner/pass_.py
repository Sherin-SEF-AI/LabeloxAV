"""The reasoning pass over one frame's fused detections, wired between fusion and the gate.

This is the seam. Everything upstream produces candidate objects; everything downstream treats them as
labels. Reasoning belongs here because it is the last moment at which a wrong label is still cheap.

What happens to one frame:

1. Build the evidence context per object, including the things the frame knows and a single detection does
   not: the other objects around it, the scene, the depth prior, the track's neighbouring observations.
2. Run Tier 1 over each. Deterministic, microseconds.
3. Escalate only the conflicted ones to Tier 2, cheapest-first so a per-frame budget is spent on the
   objects where an opinion changes the outcome.
4. Fold each verdict into the object's state and write the trace onto provenance.

The gate still decides the state. The reasoner supplies `reasoner_ok` the same way the quality reviewer
supplies `quality_ok`: a verdict that says "do not auto-accept" is honoured, and a verdict that says
"accept" only permits the gate to do what its thresholds already allow. A reasoner that could promote a
0.3 detection to accepted would be overriding the calibration rather than informing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.logging import get_logger
from core.schemas import BBox, UnifiedObject
from services.autolabel.ontology import Ontology
from services.autolabel.reasoner.evidence import EvidenceContext
from services.autolabel.reasoner.verdict import (
    ADJUDICATE,
    PERMITS_AUTO_ACCEPT,
    REJECT,
    REVIEW,
    Verdict,
    reason_about,
)

log = get_logger("reasoner_pass")


@dataclass
class FrameContext:
    """What the frame knows that a single detection does not."""

    width: int
    height: int
    scene: dict = field(default_factory=dict)
    # Metres per pixel row and focal length, from the mono-depth prior. Absent means the physics check
    # simply does not run, which is the honest degradation.
    depth_of_row: object | None = None
    focal_px: float | None = None
    # Per track id, observations in neighbouring frames: (ts_ns, bbox, class_id).
    track_history: dict[str, list[tuple[int, BBox, int]]] = field(default_factory=dict)
    # Per object index, the nearest reviewed crops: (class_name, similarity, state).
    neighbours: dict[int, list[tuple[str, float, str]]] = field(default_factory=dict)


@dataclass
class PassResult:
    verdicts: list[Verdict]
    adjudications: int = 0
    counts: dict[str, int] = field(default_factory=dict)


def _depth_for(ctx: FrameContext, obj: UnifiedObject) -> float | None:
    """Distance to the object, taken at the row where it meets the ground.

    The bottom of the box rather than its centre: an object stands on the road, and a depth read at the
    centre of a tall vehicle is the depth of its roof, which is measurably nearer and would make every
    lorry fail its own height check.
    """
    if ctx.depth_of_row is None:
        return None
    try:
        row = int(max(0, min(ctx.height - 1, obj.bbox.y2)))
        d = ctx.depth_of_row(row) if callable(ctx.depth_of_row) else None
        return float(d) if d and d > 0 else None
    except Exception:  # noqa: BLE001
        return None


def reason_frame(objects: list[UnifiedObject], onto: Ontology, frame: FrameContext,
                 *, checks: list[str] | None = None) -> list[Verdict]:
    """Tier 1 over every object in one frame."""
    verdicts: list[Verdict] = []
    for i, obj in enumerate(objects):
        history = ctx_history(frame, obj)
        ctx = EvidenceContext(
            obj=obj, onto=onto, frame_w=frame.width, frame_h=frame.height,
            others=[o for o in objects if o is not obj],
            scene=frame.scene or {},
            depth_m=_depth_for(frame, obj),
            focal_px=frame.focal_px,
            track_neighbours=history,
            neighbours=frame.neighbours.get(i, []),
        )
        verdicts.append(reason_about(ctx, only=checks or None))
    return verdicts


def ctx_history(frame: FrameContext, obj: UnifiedObject) -> list[tuple[int, BBox, int]]:
    tid = str(obj.track_id) if obj.track_id else None
    return frame.track_history.get(tid, []) if tid else []


def escalate(objects: list[UnifiedObject], verdicts: list[Verdict], onto: Ontology,
             image_bgr: np.ndarray, verifier, *, budget: int) -> int:
    """Send the conflicted objects to Tier 2, most conflicted first.

    Ordered by conflict rather than by confidence, which is the difference between this and the existing
    VLM pass. The old rule spent the budget on the least confident objects; those are often simply hard and
    a second opinion does not help. An object where two signals actively disagree is where an opinion
    changes the outcome.
    """
    from services.autolabel.reasoner.adjudicate import adjudicate, apply_adjudication

    if verifier is None or budget <= 0:
        return 0

    order = sorted((i for i, v in enumerate(verdicts) if v.decision == ADJUDICATE),
                   key=lambda i: -verdicts[i].conflict)
    used = 0
    for i in order:
        if used >= budget:
            # The rest stay as ADJUDICATE and are treated as review below: an object that needed a second
            # opinion and did not get one is not an accepted object.
            break
        obj = objects[i]
        adj = adjudicate(verifier, image_bgr, tuple(obj.bbox.as_list()), verdicts[i], onto,
                         current_class=obj.class_name)
        verdicts[i] = apply_adjudication(verdicts[i], adj, obj.class_name)
        verdicts[i].findings = [*verdicts[i].findings]
        used += 1
    return used


def apply_to_objects(objects: list[UnifiedObject], verdicts: list[Verdict],
                     *, record_trace: bool = True) -> dict[int, bool]:
    """Write each verdict onto its object and report whether the gate may auto-accept it.

    Returns a map keyed by `id(obj)`, matching how the runner already carries `quality_ok`, so the two
    signals reach the gate the same way.
    """
    ok: dict[int, bool] = {}
    for obj, verdict in zip(objects, verdicts, strict=True):
        ok[id(obj)] = verdict.decision in PERMITS_AUTO_ACCEPT
        if verdict.decision in (REJECT, REVIEW, ADJUDICATE):
            # Recorded on quality_flags as well, because the triage queue, the active-learning value score
            # and the correction loop all already read that field. A new field would have needed each of
            # them taught about it.
            reason_tags = sorted({f.check for f in verdict.findings if f.weight < 0})
            obj.provenance.quality_flags = list(
                dict.fromkeys([*obj.provenance.quality_flags, *(f"reasoner:{t}" for t in reason_tags)]))
        if record_trace:
            # Stored under its own key rather than merged into notes: attribution has to be able to read
            # each finding's weight back, and a prose note cannot be parsed into one.
            obj.provenance.reasoning = verdict.as_trace()
        if verdict.suggested_class and verdict.decision in (REVIEW, REJECT):
            obj.provenance.notes = [*obj.provenance.notes,
                                    f"reasoner suggests {verdict.suggested_class}"]
    return ok


def summarise(verdicts: list[Verdict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in verdicts:
        out[v.decision] = out.get(v.decision, 0) + 1
    return out
