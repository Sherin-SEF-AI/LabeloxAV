"""Track auto-propagation agent: label one keyframe, carry it across the clip both ways, and only bother a
human where the box actually drifts.

The existing propagate_forward walks Lucas-Kanade optical flow one direction and stops only when the flow
itself fails; every box it makes is state="annotate", so a human still reviews all of them. This agent adds
the judgement:

- bidirectional -- propagate forward AND backward from the keyframe.
- drift detection -- at each step it crops the moved box and compares its DINOv3 appearance to the source
  crop; when the similarity falls through the drift floor (the object has changed or the box slid off), it
  STOPS that direction. The frame just before is where a human re-anchors.
- confidence-graded routing -- a box that still looks like the source auto-accepts; a wobblier one routes to
  review; nothing past the drift point is created at all. So the human sees only the handful of frames the
  object genuinely changed on, not the whole clip.
- reversible -- every box it creates is recorded on one AgentRun; revert deletes them exactly.

Reuses the tested optical-flow mechanics and _gray/_clip_box from services.intelligence.propagate.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import AgentRun, Frame, Object, Track
from services.temporal import drift
from services.temporal.flow import flow_box
from services.temporal.flow import gray as _gray

log = get_logger("agent.propagate")

# The shared, guarded flow step (services/temporal/flow.py), under the name the walk below uses.
_flow_box = flow_box


# The appearance primitives live in services/temporal/drift.py now, because this check is also what says
# whether a box already in the corpus is still on its object, and that pass needs the same three
# functions. Aliased rather than renamed here: the walk below reads better with these names, and they are
# the names the docstring at the top of this file uses.
_crop = drift.crop_box
_appearance = drift.cosine_to
_source_vec = drift.encode_crop


async def _neighbors(db, session_id, ts_ns, *, forward: bool, span: int, cam_id: str | None = None):
    q = select(Frame).where(Frame.session_id == session_id)
    if cam_id is not None:
        # The source camera only: flow from one view into another walks the box onto something else.
        q = q.where(Frame.cam_id == cam_id)
    if forward:
        q = q.where(Frame.ts_ns > ts_ns).order_by(Frame.ts_ns.asc())
    else:
        q = q.where(Frame.ts_ns < ts_ns).order_by(Frame.ts_ns.desc())
    return list((await db.execute(q.limit(span))).scalars().all())


async def _walk(db, store, src, src_frame, ref_vec, *, forward, span, drift, high, size_tol):
    """Walk one direction from the keyframe, producing a step per frame until drift/flow-loss. Read-only."""
    frames = await _neighbors(db, src_frame.session_id, src_frame.ts_ns, forward=forward, span=span,
                              cam_id=src_frame.cam_id)
    steps: list[dict] = []
    prev_gray = _gray(store, src_frame.img_uri)
    box = list(src.bbox)
    src_area = max(1.0, (src.bbox[2] - src.bbox[0]) * (src.bbox[3] - src.bbox[1]))
    for dist, nf in enumerate(frames, start=1):
        cur_gray = _gray(store, nf.img_uri)
        if prev_gray is None or cur_gray is None:
            break
        moved = _flow_box(prev_gray, cur_gray, box)
        if moved is None:
            break
        area = max(1.0, (moved[2] - moved[0]) * (moved[3] - moved[1]))
        ratio = area / src_area
        sim = _appearance(store, nf.img_uri, moved, ref_vec)
        # geometry sanity: the box should not balloon or collapse relative to the keyframe
        geom_ok = (1.0 / size_tol) <= ratio <= size_tol
        drifted = (sim is not None and sim < drift) or not geom_ok
        base = max(0.4, 0.92 - 0.02 * dist)
        conf = min(base, sim) if sim is not None else base
        if drifted:
            steps.append({"frame_id": str(nf.frame_id), "ts_ns": int(nf.ts_ns), "box": [float(v) for v in moved],
                          "direction": "fwd" if forward else "bwd", "distance": dist,
                          "appearance": round(sim, 3) if sim is not None else None, "conf": round(conf, 3),
                          "action": "stop", "reason": "appearance drift" if (sim is not None and sim < drift) else "box size drift"})
            break
        action = "auto_accept" if (sim is None and dist <= 3) or (sim is not None and sim >= high) else "review"
        steps.append({"frame_id": str(nf.frame_id), "ts_ns": int(nf.ts_ns), "box": [float(v) for v in moved],
                      "direction": "fwd" if forward else "bwd", "distance": dist,
                      "appearance": round(sim, 3) if sim is not None else None, "conf": round(conf, 3),
                      "action": action})
        box = moved
        prev_gray = cur_gray
    return steps


async def plan_propagate(db: AsyncSession, object_id: uuid.UUID, *, span: int = 24, drift: float = 0.62,
                         high: float = 0.80, size_tol: float = 3.0) -> dict:
    """Dry-run: what the agent would propagate from this keyframe, both ways, with drift stops. No writes."""
    src = await db.get(Object, object_id)
    if src is None:
        raise ValueError("object not found")
    src_frame = await db.get(Frame, src.frame_id)
    store = get_object_store()
    ref_vec = _source_vec(store, src_frame.img_uri, src.bbox)
    fwd = await _walk(db, store, src, src_frame, ref_vec, forward=True, span=span, drift=drift, high=high, size_tol=size_tol)
    bwd = await _walk(db, store, src, src_frame, ref_vec, forward=False, span=span, drift=drift, high=high, size_tol=size_tol)
    steps = fwd + bwd
    created = [s for s in steps if s["action"] != "stop"]
    counts = {"total_steps": len(steps), "auto_accept": sum(s["action"] == "auto_accept" for s in created),
              "review": sum(s["action"] == "review" for s in created),
              "stops": sum(s["action"] == "stop" for s in steps), "appearance_used": ref_vec is not None}
    return {"object_id": str(object_id), "class_id": src.class_id, "counts": counts,
            "forward": len([s for s in fwd if s["action"] != "stop"]),
            "backward": len([s for s in bwd if s["action"] != "stop"]), "steps": steps}


async def commit_propagate(db: AsyncSession, object_id: uuid.UUID, *, span: int = 24, drift: float = 0.62,
                           high: float = 0.80, size_tol: float = 3.0, created_by: str | None = None) -> dict:
    """Propagate and persist the boxes as one reversible AgentRun. Only machine boxes are created; the
    source keyframe is never touched."""
    plan = await plan_propagate(db, object_id, span=span, drift=drift, high=high, size_tol=size_tol)
    src = await db.get(Object, object_id)
    src_frame = await db.get(Frame, src.frame_id)

    # ensure a track to thread the boxes onto
    track_id = src.track_id
    if track_id is None:
        tr = Track(session_id=src_frame.session_id, class_id=src.class_id,
                   first_ts_ns=src_frame.ts_ns, last_ts_ns=src_frame.ts_ns)
        db.add(tr)
        await db.flush()
        track_id = tr.track_id
        src.track_id = track_id

    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    ts_seen = []
    for s in plan["steps"]:
        if s["action"] == "stop":
            continue
        oid = uuid.uuid4()
        db.add(Object(
            object_id=oid, frame_id=uuid.UUID(s["frame_id"]), track_id=track_id, class_id=src.class_id,
            bbox=s["box"], conf=float(s["conf"]), source="propagated", state=s["action"],
            interp_source="propagated", attrs={},
            provenance={"propagated_from": str(object_id), "method": "optical-flow-agent",
                        "direction": s["direction"], "distance": s["distance"],
                        "appearance": s["appearance"], "agent_run_id": str(run_id)},
        ))
        changes[str(oid)] = {"created": True, "track_id": str(track_id)}
        ts_seen.append(s["ts_ns"])

    if ts_seen:
        tr = await db.get(Track, track_id)
        tr.first_ts_ns = min(tr.first_ts_ns or src_frame.ts_ns, min(ts_seen), src_frame.ts_ns)
        tr.last_ts_ns = max(tr.last_ts_ns or src_frame.ts_ns, max(ts_seen), src_frame.ts_ns)

    db.add(AgentRun(run_id=run_id, kind="propagate", scope={"object_id": str(object_id), "track_id": str(track_id)},
                    status="committed", policy={"span": span, "drift": drift, "high": high},
                    counts=plan["counts"], changes=changes, critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.propagate.commit", object_id=str(object_id), run_id=str(run_id), created=len(changes))
    return {"run_id": str(run_id), "object_id": str(object_id), "track_id": str(track_id),
            "created": len(changes), "counts": plan["counts"]}
