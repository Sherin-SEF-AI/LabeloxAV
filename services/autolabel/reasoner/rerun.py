"""Reason over objects that already exist, without re-detecting them.

Two situations need this, and both are the ordinary case rather than the exception.

The first is the corpus you already have. Forty thousand frames were annotated before this layer existed,
and re-running detection over them costs GPU hours to arrive at the same boxes. The reasoning is the part
that is new, and it can be applied to the boxes as they stand.

The second is a changed prior. Somebody corrects a height band or adds a confusable pair, and the question
immediately after is "what would that have caught". Answering it by re-annotating is both expensive and
wrong, because it would change the boxes too and confound the comparison.

`apply` is off by default. Seeing what the reasoner would do and doing it are different acts, and a rerun
that silently demoted ten thousand accepted labels because a prior was mistyped is exactly the kind of
thing that should require somebody to have meant it.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.schemas import BBox, PathProposal, Provenance, UnifiedObject

log = get_logger("reasoner_rerun")


def _to_unified(row, class_name: str) -> UnifiedObject:
    """Rebuild the in-memory object from its stored row.

    The provenance is reconstructed rather than passed as a blob because the collectors read typed fields
    from it (proposals, agreement, entropy, mask_box_disagree), and a dict would make every one of them
    defensive about shapes it should be able to rely on.
    """
    prov_raw = dict(row.provenance or {})
    proposals = []
    for p in (prov_raw.get("proposals") or []):
        try:
            proposals.append(PathProposal(
                path=str(p.get("path", "?")), class_name=p.get("class_name"), conf=p.get("conf"),
                verdict=str(p.get("verdict", "proposed")),
                model_version=str(p.get("model_version", "unknown"))))
        except Exception:  # noqa: BLE001 - a malformed historical proposal is skipped, not fatal
            continue

    bbox = [float(v) for v in (row.bbox or [0, 0, 1, 1])]
    return UnifiedObject(
        object_id=row.object_id, frame_id=row.frame_id, track_id=row.track_id,
        class_id=row.class_id, class_name=class_name,
        bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        conf=float(row.conf or 0.0),
        provenance=Provenance(
            proposals=proposals,
            agreement=bool(prov_raw.get("agreement")),
            mask_box_disagree=bool(prov_raw.get("mask_box_disagree")),
            entropy=prov_raw.get("entropy"),
            quality_flags=list(prov_raw.get("quality_flags") or []),
            notes=list(prov_raw.get("notes") or []),
        ))


async def rerun_session(db: AsyncSession, session_id: str, *, limit: int = 500,
                        apply: bool = False) -> dict:
    """Reason over one session's existing objects and report what would change.

    Human decisions are never overwritten, whatever `apply` says. An object a person accepted or rejected
    is settled, and a machine revisiting it would undo exactly the work the loop exists to collect.
    """
    from db.models import Frame, Object
    from services.autolabel.ontology import get_ontology
    from services.autolabel.reasoner.pass_ import (
        FrameContext,
        apply_to_objects,
        reason_frame,
        summarise,
    )

    onto = get_ontology()
    rows = (await db.execute(
        select(Object, Frame.frame_id, Frame.ts_ns, Frame.width, Frame.height, Frame.scene)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == uuid.UUID(session_id))
        .order_by(Frame.ts_ns)
        .limit(limit))).all()
    if not rows:
        return {"session_id": session_id, "objects": 0,
                "detail": "the session has no objects to reason about"}

    # Grouped by frame, because half the checks are about an object's neighbours and its scene, and an
    # object reasoned about alone loses exactly the context that makes the layer worth having.
    by_frame: dict = {}
    for obj, fid, ts, w, h, scene in rows:
        by_frame.setdefault(str(fid), {"ts": int(ts or 0), "w": int(w or 0), "h": int(h or 0),
                                       "scene": scene or {}, "rows": []})["rows"].append(obj)

    # Track history is built as frames are walked, in timestamp order, so the temporal check sees the same
    # thing it would have seen live.
    track_history: dict[str, list] = {}
    counts: dict[str, int] = {}
    changes: list[dict] = []
    applied = 0
    skipped_human = 0
    auto_accepted = 0

    for fid in sorted(by_frame, key=lambda k: by_frame[k]["ts"]):
        f = by_frame[fid]
        unified = [_to_unified(r, onto.by_id(r.class_id).name) for r in f["rows"]]
        verdicts = reason_frame(unified, onto,
                                FrameContext(width=f["w"], height=f["h"], scene=f["scene"],
                                             track_history=track_history))
        ok = apply_to_objects(unified, verdicts, record_trace=True)
        for decision, n in summarise(verdicts).items():
            counts[decision] = counts.get(decision, 0) + n

        for row, u, verdict in zip(f["rows"], unified, verdicts, strict=True):
            if row.state == "auto_accept":
                auto_accepted += 1
            would_demote = not ok[id(u)] and row.state in ("auto_accept", "accepted")
            if would_demote:
                changes.append({
                    "object_id": str(row.object_id), "frame_id": fid,
                    "class_name": u.class_name, "state": row.state,
                    "decision": verdict.decision,
                    "suggested_class": verdict.suggested_class,
                    "reasons": verdict.reasons[:3]})

            if not apply:
                continue
            if row.source == "human" or row.state in ("accepted", "rejected"):
                # Settled by a person. Revisiting it would undo the work the loop exists to collect.
                skipped_human += 1
                continue
            row.provenance = {**(row.provenance or {}), **{
                "reasoning": u.provenance.reasoning,
                "quality_flags": u.provenance.quality_flags,
                "notes": u.provenance.notes}}
            if would_demote:
                row.state = "review"
                applied += 1

        for u in unified:
            if u.track_id:
                hist = track_history.setdefault(str(u.track_id), [])
                hist.append((f["ts"], u.bbox, u.class_id))
                if len(hist) > 6:
                    del hist[:-6]

    if apply:
        await db.commit()

    log.info("reasoner.rerun", session=session_id, objects=len(rows),
             would_demote=len(changes), applied=applied, dry_run=not apply)
    return {
        "session_id": session_id, "objects": len(rows), "frames": len(by_frame),
        "decisions": counts,
        "would_demote": len(changes),
        "auto_accepted": auto_accepted,
        # The number that actually matters, and it is deliberately not "demotions over all objects".
        # Most objects in a session were never auto-accepted, so dividing by all of them understates the
        # intervention by an order of magnitude: on a real session it read 0.8% when the reasoner was in
        # fact disagreeing with 15% of the labels that were going into training unreviewed.
        "demote_rate_of_auto_accepted": (round(len(changes) / auto_accepted, 4)
                                         if auto_accepted else None),
        "demote_rate_of_all": round(len(changes) / len(rows), 4),
        "applied": applied,
        "skipped_human_decisions": skipped_human,
        "dry_run": not apply,
        "examples": changes[:25],
        "truncated_examples": max(0, len(changes) - 25),
    }
