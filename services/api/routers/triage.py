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


# How loudly each reason should read. A defect and a piece of context are not the same news, and rendering
# them identically is what made a screen of flags unsortable by eye.
#
#   high    something is probably wrong with this object
#   medium  worth a look, and the reason the ranker put it here
#   low     context; true, but not on its own a call to act
#
# Severity lives here rather than in the client because the evidence that decides it lives here. The web app
# previously recovered it by running regexes over the joined prose, which put the taxonomy in two places and
# meant a wording change silently restyled the queue.
FLAG_SEVERITY = {
    "mask_box": "high",
    "class_conflict": "high",
    "low_conf": "medium",
    "rare_class": "low",
    "review_band": "low",
}

# Named for what the reader is being asked to check, not for the comparison that produced them. "mask != box"
# is a column name; the annotator's question is whether the outline fits the box.
FLAG_LABEL = {
    "mask_box": "Outline off box",
    "class_conflict": "Class conflict",
    "low_conf": "Low confidence",
    "rare_class": "Rare class",
    "review_band": "In review band",
}

# The prose kept on `why`, so existing readers of that field are unaffected.
FLAG_PROSE = {
    "mask_box": "mask != box",
    "class_conflict": "class conflict",
    "rare_class": "rare class",
    "low_conf": "low conf",
    "review_band": "review band",
}

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

# The score below which a detection counts as unconfident. Strictly below: 0.6 is the floor and is not itself
# low, which is the sort of boundary a later `<=` moves without anybody noticing.
LOW_CONF_FLOOR = 0.6


def _why_and_priority(obj: Object, onto) -> tuple[str, float, list[dict]]:
    """The reasons this object is in the queue, its rank, and those reasons as sorted structured flags."""
    c = onto.by_id(obj.class_id)
    prov = obj.provenance or {}
    rare = c.india or c.l1 == "fallback"
    mask_box = bool(prov.get("mask_box_disagree"))
    conflict = sum(1 for p in prov.get("proposals", []) if p.get("verdict") == "overruled") > 0 and len(prov.get("proposals", [])) > 1

    codes = []
    if mask_box:
        codes.append("mask_box")
    if conflict:
        codes.append("class_conflict")
    if rare:
        codes.append("rare_class")
    if obj.conf < LOW_CONF_FLOOR:
        codes.append("low_conf")
    # An object can sit in the review band with nothing wrong with it. That is not a defect and must not be
    # dressed as one, so it gets its own low flag rather than an empty list the client has to interpret.
    if not codes:
        codes.append("review_band")

    codes.sort(key=lambda code: _SEVERITY_RANK[FLAG_SEVERITY[code]])
    flags = [{"code": code, "label": FLAG_LABEL[code], "severity": FLAG_SEVERITY[code]} for code in codes]
    why = ", ".join(FLAG_PROSE[code] for code in codes)

    uncertainty = 1.0 - obj.conf
    rarity = 2.0 if rare else 1.0
    boost = 1.0 + (0.5 if mask_box else 0.0) + (0.5 if conflict else 0.0)
    return why, round(uncertainty * rarity * boost, 4), flags


@router.get("/triage", response_model=list[TriageRow])
async def triage(
    db: AsyncSession = Depends(db_session),
    states: str = Query("review,annotate"),
    session_id: str | None = None,
    klass: str | None = None,
    city: str | None = None,
    flywheel: str | None = None,
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
    if flywheel:
        # the worklist a flywheel cycle dispatched: objects it stamped with this cycle id
        stmt = stmt.where(Object.provenance["flywheel"]["cycle_id"].astext == flywheel)
    stmt = stmt.limit(max(limit * 3, limit))  # over-fetch, rank, then trim

    rows = (await db.execute(stmt)).all()
    out: list[TriageRow] = []
    for obj, sid in rows:
        why, priority, flags = _why_and_priority(obj, onto)
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
                flags=flags,
                priority=priority,
                source=obj.source,
                import_format=(obj.provenance or {}).get("import_format"),
            )
        )
    out.sort(key=lambda r: r.priority, reverse=True)
    return out[:limit]
