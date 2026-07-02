"""Policy compliance engine: declarative annotation-guideline rules checked corpus-wide. Each rule is a
small, explainable predicate over an object's geometry, class, and attributes; a violation becomes a ranked
policy_violation ErrorCandidate in the fix queue. Rules catch the mechanical mistakes the model-based
detectors do not: specks too small to be a real object, a pedestrian box wider than tall, attributes that
are invalid or not applicable to the class, and duplicate boxes on the same frame. Adding a guideline is
one entry in RULES.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("ed.policy")

_MIN_AREA_FRAC = 0.00012   # boxes below this fraction of the frame are almost always noise
_DUP_IOU = 0.85            # same-class boxes overlapping more than this are redundant


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 1e-6 else 0.0


def check_object(o, others, onto, w: int, h: int) -> list[tuple[str, float, str]]:
    """Return (rule, score, reason) for each guideline this object breaks."""
    out: list[tuple[str, float, str]] = []
    x1, y1, x2, y2 = (float(v) for v in o.bbox)
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    try:
        c = onto.by_id(int(o.class_id))
        l1, name = c.l1, c.name
    except Exception:  # noqa: BLE001
        l1, name = "", str(o.class_id)

    if bw * bh < _MIN_AREA_FRAC * (w * h):
        out.append(("min_box_size", 0.6, f"box is {int(bw)}x{int(bh)} px, below the minimum size"))

    if bh > 0 and bw > 0:
        ar = bw / bh
        if l1 == "vru" and ar > 1.6:
            out.append(("degenerate_aspect", 0.7, f"{name} box is wider than tall (aspect {ar:.1f})"))
        elif l1 in ("two_wheeler", "three_wheeler", "four_wheeler", "heavy") and (ar > 6.0 or ar < 0.15):
            out.append(("degenerate_aspect", 0.65, f"{name} box aspect {ar:.1f} is implausible"))

    errs = onto.validate_attrs(o.attrs or {}, int(o.class_id))
    if errs:
        out.append(("attr_validity", 0.75, "; ".join(errs[:3])))

    for ob in others:
        if ob.object_id == o.object_id or int(ob.class_id) != int(o.class_id):
            continue
        if _iou([x1, y1, x2, y2], [float(v) for v in ob.bbox]) > _DUP_IOU:
            out.append(("duplicate_box", 0.7, f"overlaps another {name} box (IoU > {_DUP_IOU})"))
            break
    return out


async def detect_policy_violations(db: AsyncSession, session_id: str | None = None, *,
                                   limit_frames: int | None = None) -> list[dict]:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    q = select(Object.frame_id).where(Object.source != "human").group_by(Object.frame_id)
    if session_id:
        q = select(Object.frame_id).join(Frame, Frame.frame_id == Object.frame_id).where(
            Frame.session_id == UUID(session_id), Object.source != "human").group_by(Object.frame_id)
    if limit_frames:
        q = q.limit(limit_frames)
    frame_ids = list((await db.execute(q)).scalars().all())

    out: list[dict] = []
    for fid in frame_ids:
        frame = await db.get(Frame, fid)
        if frame is None:
            continue
        objs = (await db.execute(select(Object).where(Object.frame_id == fid, Object.source != "human"))).scalars().all()
        for o in objs:
            for rule, score, reason in check_object(o, objs, onto, frame.width, frame.height):
                out.append({"object_id": str(o.object_id), "kind": "policy_violation", "score": round(score, 4),
                            "proposed_label": None, "detail": {"rule": rule, "reason": reason}})
    log.info("ed.policy.done", frames=len(frame_ids), violations=len(out), scope=session_id or "corpus")
    return out
