"""Triage queue: a ranked work list, not a folder tree. Ordered by an active-learning priority
(uncertainty x class-rarity, with conflict boosts), each row surfacing its uncertainty reason."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object
from db.models import Session as DbSession
from services.api.deps import TriageRow, db_session
from services.autolabel.ontology import get_ontology

router = APIRouter()


def _why_and_priority(obj: Object, onto) -> tuple[str, float]:
    c = onto.by_id(obj.class_id)
    prov = obj.provenance or {}
    rare = c.india or c.l1 == "fallback"
    mask_box = bool(prov.get("mask_box_disagree"))
    conflict = sum(1 for p in prov.get("proposals", []) if p.get("verdict") == "overruled") > 0 and len(prov.get("proposals", [])) > 1

    reasons = []
    if mask_box:
        reasons.append("mask != box")
    if conflict:
        reasons.append("class conflict")
    if rare:
        reasons.append("rare class")
    if obj.conf < 0.6:
        reasons.append("low conf")
    why = ", ".join(reasons) or "review band"

    uncertainty = 1.0 - obj.conf
    rarity = 2.0 if rare else 1.0
    boost = 1.0 + (0.5 if mask_box else 0.0) + (0.5 if conflict else 0.0)
    return why, round(uncertainty * rarity * boost, 4)


@router.get("/triage", response_model=list[TriageRow])
async def triage(
    db: AsyncSession = Depends(db_session),
    states: str = Query("review,annotate"),
    session_id: str | None = None,
    klass: str | None = None,
    city: str | None = None,
    limit: int = 200,
):
    limit = min(max(limit, 1), 1000)
    onto = get_ontology()
    state_list = [s.strip() for s in states.split(",") if s.strip()]
    stmt = (
        select(Object, Frame.session_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .join(DbSession, Frame.session_id == DbSession.session_id)
        .where(Object.state.in_(state_list))
    )
    if session_id:
        stmt = stmt.where(Frame.session_id == UUID(session_id))
    if city:
        stmt = stmt.where(DbSession.city == city)
    if klass:
        stmt = stmt.where(Object.class_id == onto.by_name(klass).id)
    stmt = stmt.limit(max(limit * 3, limit))  # over-fetch, rank, then trim

    rows = (await db.execute(stmt)).all()
    out: list[TriageRow] = []
    for obj, sid in rows:
        why, priority = _why_and_priority(obj, onto)
        out.append(
            TriageRow(
                object_id=str(obj.object_id),
                frame_id=str(obj.frame_id),
                session_id=str(sid),
                class_id=obj.class_id,
                class_name=onto.by_id(obj.class_id).name,
                conf=obj.conf,
                state=obj.state,
                why=why,
                priority=priority,
                source=obj.source,
                import_format=(obj.provenance or {}).get("import_format"),
            )
        )
    out.sort(key=lambda r: r.priority, reverse=True)
    return out[:limit]
