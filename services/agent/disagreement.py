"""Champion-vs-challenger disagreement mining: the frames where the models disagree are the highest-value
frames to label and the earliest warning of a regression. Rather than re-run a challenger live, it reads
the disagreement the fusion pipeline already recorded -- each object's provenance keeps every detection
path's class vote (the champion YOLO and the open-vocab challenger), so an object whose paths voted
different classes is a live disagreement. It surfaces the strongest ones (a confident dissenting vote) into
the ScenarioCandidate queue as kind model_disagreement, ranked by how confidently the paths disagree.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ScenarioCandidate

log = get_logger("agent.disagreement")


def _disagreement(obj) -> tuple[float, list[str]] | None:
    """(strength, conflicting_classes) if the object's detection paths voted different classes, else None."""
    prov = obj.provenance or {}
    props = prov.get("proposals") or []
    votes = [(p.get("class_name"), float(p.get("conf") or 0.0)) for p in props if p.get("class_name")]
    classes = {c for c, _ in votes}
    if len(classes) < 2:
        return None
    accepted = None
    try:
        from services.autolabel.ontology import get_ontology
        accepted = get_ontology().by_id(int(obj.class_id)).name
    except Exception:  # noqa: BLE001
        pass
    dissent = max((conf for cls, conf in votes if cls != accepted), default=0.0)
    return dissent, sorted(classes)


async def mine_disagreements(db: AsyncSession, session_id: str | None = None, *, min_dissent: float = 0.15) -> dict:
    """Mine model-disagreement objects into the ScenarioCandidate queue. Idempotent for the kind."""
    q = (select(Object, Frame.session_id, Frame.frame_id)
         .join(Frame, Frame.frame_id == Object.frame_id).where(Object.source != "human"))
    if session_id:
        q = q.where(Frame.session_id == UUID(session_id))
    rows = (await db.execute(q)).all()

    best: dict[str, dict] = {}  # per frame, keep the strongest disagreement
    for obj, sid, fid in rows:
        d = _disagreement(obj)
        if d is None or d[0] < min_dissent:
            continue
        strength, classes = d
        key = str(fid)
        if key not in best or strength > best[key]["score"]:
            best[key] = {"session_id": sid, "frame_id": fid, "score": round(float(strength), 3),
                         "tag": f"paths disagree: {' vs '.join(classes[:3])}"}

    del_q = delete(ScenarioCandidate).where(ScenarioCandidate.kind == "model_disagreement", ScenarioCandidate.state == "pending")
    if session_id:
        del_q = del_q.where(ScenarioCandidate.session_id == UUID(session_id))
    await db.execute(del_q)
    for c in best.values():
        db.add(ScenarioCandidate(session_id=c["session_id"], frame_id=c["frame_id"], kind="model_disagreement",
                                 score=c["score"], state="pending", tag=c["tag"]))
    await db.commit()

    top = sorted(best.values(), key=lambda c: -c["score"])[:10]
    log.info("agent.disagreement", persisted=len(best), scope=session_id or "corpus")
    return {"persisted": len(best),
            "top": [{"score": c["score"], "tag": c["tag"], "frame_id": str(c["frame_id"])} for c in top]}
