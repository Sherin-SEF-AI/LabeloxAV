"""Filling the frames between anchors on a track, and refusing to when the anchors are not one object.

Between anchors, boxes are interpolated in centre-and-size space with a shape-preserving spline and marked
`source=interpolated` with an `interp_source`, so provenance shows they are machine-filled.

**The refusal is the important part.** An earlier corpus-wide fill created 137,913 objects that judge at
0.209 precision against 0.603 for real detections. The arithmetic here was never the problem - printing a
track's boxes in time order shows the fills landing exactly where they should. The problem is that most
anchor pairs were not the same object: only 20.9% of the holes that were filled had endpoints within a
plausible displacement, scale and class of each other, and 0.209 of the objects produced were right. So
every hole now passes `services/temporal/gap_gate.py::same_object` before anything is written, and a run
reports what it declined and why rather than only what it made.

Two anchor policies, because the two callers want different things and conflating them is how the corpus
fill went wrong. `keyframe` anchors on human-verified or explicitly marked boxes, which is what the editor
means by interpolating between keyframes - but only 179 of 11,406 tracks have two of those, so it is useless
for a backfill. `detection` anchors on detector output. Neither ever anchors on `interpolated` or
`propagated` boxes: the old implementation treated every object as an anchor, so its own output re-anchored
the next run and the errors compounded.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import delete, or_, select

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, Track
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("interpolate")


# What may serve as an anchor under each policy. `interpolated` and `propagated` are absent from both by
# construction: a fill anchored on a previous fill compounds its own error, which is what the corpus-wide
# run did on every re-run.
ANCHOR_SOURCES = {
    "keyframe": None,                                          # human / is_keyframe, resolved below
    "detection": ("fused", "auto_accept", "imported", "human"),
}


async def _anchors(db, track_id: UUID, policy: str = "keyframe"):
    """Anchor objects on a track under the named policy, ordered by time."""
    q = (select(Object, Frame.ts_ns).join(Frame, Frame.frame_id == Object.frame_id)
         .where(Object.track_id == track_id))
    if policy == "keyframe":
        q = q.where(or_(Object.is_keyframe.is_(True), Object.source == "human"))
    else:
        sources = ANCHOR_SOURCES.get(policy)
        if sources is None:
            raise ValueError(f"unknown anchor policy '{policy}'")
        q = q.where(Object.source.in_(sources))
    return (await db.execute(q.order_by(Frame.ts_ns))).all()


def _to_cxcywh(box: np.ndarray) -> np.ndarray:
    """Corner boxes (N,4 as x1,y1,x2,y2) -> center+size (N,4 as cx,cy,w,h). We interpolate motion and scale
    separately: a car approaching the camera grows in a way that oscillates badly if you spline the corners
    independently, but is smooth and monotone in width/height."""
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=1)


def build_box_interpolator(kf_ts: list[int], kf_box: np.ndarray, method: str):
    """Return (box_at, src). `box_at(ts)` gives an [x1,y1,x2,y2] box for any ts inside the keyframe span.

    method='cubic' uses a shape-preserving monotone spline (PCHIP) on center and size. Unlike an ordinary
    cubic it does not overshoot between anchors (no Runge wobble, no box that briefly balloons or inverts),
    while still curving through the acceleration a straight line would miss. Falls back to linear with <3
    keyframes or if SciPy is unavailable.
    """
    ts = np.asarray(kf_ts, dtype=float)
    cc = _to_cxcywh(np.asarray(kf_box, dtype=float))
    # collapse duplicate timestamps (two keyframes on the same frame) so the spline sees a strictly increasing grid
    uniq_ts, idx = np.unique(ts, return_index=True)
    ts, cc = uniq_ts, cc[idx]

    fns = None
    src = "linear"
    if method in ("cubic", "pchip", "spline") and len(ts) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            fns = [PchipInterpolator(ts, cc[:, i], extrapolate=True) for i in range(4)]
            src = "cubic"
        except Exception:  # noqa: BLE001 - SciPy missing/edge case: degrade to linear rather than fail the fill
            fns = None

    def box_at(t: float) -> list[float]:
        if fns is not None:
            cx, cy, w, h = (float(fns[i](t)) for i in range(4))
        else:
            cx, cy, w, h = (float(np.interp(t, ts, cc[:, i])) for i in range(4))
        w, h = max(1.0, w), max(1.0, h)   # a spline must never emit a zero-area or inverted box
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    return box_at, src


def _interp_conf(ts: float, kf_ts: list[int]) -> float:
    """Confidence that decays with temporal distance from the nearest keyframe: an anchor-adjacent fill is
    trustworthy, a fill in the middle of a long gap is not, so it earns a lower conf and lands in review."""
    left = max((k for k in kf_ts if k <= ts), default=kf_ts[0])
    right = min((k for k in kf_ts if k >= ts), default=kf_ts[-1])
    span = right - left
    if span <= 0:
        return 0.55
    t = (ts - left) / span                       # 0 at left anchor, 1 at right anchor
    closeness = 1.0 - abs(2 * t - 1)             # 0 at an anchor, 1 at the gap midpoint
    return round(0.55 - 0.25 * closeness, 3)     # 0.55 next to an anchor, 0.30 mid-gap


async def interpolate_track_keyframed(track_id: UUID, method: str = "linear", lo_ts: int | None = None,
                                      hi_ts: int | None = None, *, anchor_policy: str = "keyframe",
                                      run_id: UUID | None = None, gate: bool = True) -> dict:
    """Fill frames between anchors with interpolated boxes, skipping holes whose anchors are not one object.

    `anchor_policy` selects what counts as an anchor - see ANCHOR_SOURCES. `run_id` makes the fill revertible:
    every created object is stamped with it and recorded as `{"created": True}` on the run, which is the
    shape `services/agent/runs.py::revert_run` already deletes. `gate=False` is for the editor's explicit
    "interpolate between the two boxes I just drew", where a person has already asserted they are one object.

    Returns per-hole accounting, including `refused` keyed by reason. A fill that reports only what it
    created cannot be distinguished from one that created the wrong thing, which is how the last one passed.
    """
    from services.domain import active_pack
    from services.temporal.gap_gate import same_object

    cliques = active_pack().cliques
    maker = get_sessionmaker()
    async with maker() as db:
        tr = await db.get(Track, track_id)
        if tr is None:
            return {"created": 0, "reason": "track not found"}
        anchors = await _anchors(db, track_id, anchor_policy)
        if len(anchors) < 2:
            return {"created": 0, "reason": f"need at least 2 {anchor_policy} anchors"}

        kf_ts = [ts for _, ts in anchors]
        kf_box = np.asarray([list(o.bbox) for o, _ in anchors], dtype=float)
        class_id = anchors[0][0].class_id
        a, b = (lo_ts if lo_ts is not None else kf_ts[0]), (hi_ts if hi_ts is not None else kf_ts[-1])

        # Confine interpolation to the track's own camera. A rig session has frames from several cameras at
        # overlapping timestamps; without this filter the fill would create boxes on every camera's frames,
        # poisoning views the track was never in.
        anchor_frame = await db.get(Frame, anchors[0][0].frame_id)
        cam_id = anchor_frame.cam_id if anchor_frame else None

        fq = (select(Frame.frame_id, Frame.ts_ns)
              .where(Frame.session_id == tr.session_id, Frame.ts_ns > a, Frame.ts_ns < b))
        if cam_id is not None:
            fq = fq.where(Frame.cam_id == cam_id)
        frames = (await db.execute(fq.order_by(Frame.ts_ns))).all()
        # clear existing machine-filled boxes on this track in the segment (idempotent re-interpolation)
        seg_fids = [fid for fid, _ in frames]
        if seg_fids:
            await db.execute(delete(Object).where(
                Object.track_id == track_id, Object.source == "interpolated", Object.frame_id.in_(seg_fids)))

        box_at, src = build_box_interpolator(kf_ts, kf_box, method)

        # Anchor lookup by class name and by index, for the gate.
        onto = get_ontology()
        name_of = {}
        for o, ts in anchors:
            try:
                name_of[ts] = onto.by_id(int(o.class_id)).name
            except KeyError:
                name_of[ts] = str(o.class_id)
        width = float(anchor_frame.width) if anchor_frame and anchor_frame.width else 1920.0

        # Group the holes by the anchor pair that brackets them, so the gate is asked once per hole rather
        # than once per frame and a refusal drops the whole hole rather than half of it.
        kf_set = set(kf_ts)
        holes: dict[tuple[int, int], list] = {}
        for fid, ts in frames:
            if ts in kf_set:
                continue
            lo = max((k for k in kf_ts if k <= ts), default=None)
            hi = min((k for k in kf_ts if k >= ts), default=None)
            if lo is None or hi is None:
                continue
            holes.setdefault((lo, hi), []).append((fid, ts))

        by_ts = {ts: o for o, ts in anchors}
        created = 0
        refused: dict[str, int] = {}
        refused_frames = 0
        for (lo, hi), members in sorted(holes.items()):
            if gate:
                res = same_object(by_ts[lo].bbox, by_ts[hi].bbox, name_of[lo], name_of[hi],
                                  frame_width=width, gap_frames=len(members), cliques=cliques)
                if not res.ok:
                    refused[res.reason] = refused.get(res.reason, 0) + 1
                    refused_frames += len(members)
                    continue
            for fid, ts in members:
                conf = _interp_conf(float(ts), kf_ts)
                prov = {"method": "interpolate", "interp_source": src, "conf_by_gap": conf,
                        "gap_frames": len(members)}
                if run_id is not None:
                    prov["agent_run_id"] = str(run_id)
                db.add(Object(frame_id=fid, track_id=track_id, class_id=class_id, bbox=box_at(float(ts)),
                              conf=conf, source="interpolated", state="annotate", interp_source=src,
                              provenance=prov))
                created += 1

        if run_id is not None and created:
            # Stamped in the same transaction as the objects. Recording the run afterwards would leave a
            # crash between the two with rows nothing can revert, which is the one thing this exists for.
            await db.flush()
            made = (await db.execute(select(Object.object_id).where(
                Object.track_id == track_id, Object.source == "interpolated",
                Object.provenance["agent_run_id"].astext == str(run_id)))).scalars().all()
            run = await db.get(AgentRun, run_id)
            if run is not None:
                run.changes = {**(run.changes or {}), **{str(oid): {"created": True} for oid in made}}
        await db.commit()

    out = {"track_id": str(track_id), "created": created, "method": src, "anchors": len(kf_ts),
           "anchor_policy": anchor_policy, "holes": len(holes), "refused": refused,
           "refused_frames": refused_frames}
    log.info("interpolate.done", **{k: v for k, v in out.items() if k != "refused"})
    return out
