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
from services.api.project_guard import caller_hidden_projects
from services.autolabel.ontology import get_ontology
from services.project_access import scope_visible_sessions
from services.queue_order import (
    LOW_CONF_FLOOR,
    class_conflict,
    mask_box_disagrees,
    priority_of,
    priority_sql,
    rare_class_ids,
)

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
    "control_sample": "medium",
    "mask_box": "high",
    "class_conflict": "high",
    "low_conf": "medium",
    "rare_class": "low",
    "review_band": "low",
    "settlement_sample": "medium",
    "settlement_spot": "medium",
}

# Named for what the reader is being asked to check, not for the comparison that produced them. "mask != box"
# is a column name; the annotator's question is whether the outline fits the box.
FLAG_LABEL = {
    "control_sample": "Control sample",
    "mask_box": "Outline off box",
    "class_conflict": "Class conflict",
    "low_conf": "Low confidence",
    "rare_class": "Rare class",
    "review_band": "In review band",
    "settlement_sample": "Settlement sample",
    "settlement_spot": "Settlement spot check",
}

# The prose kept on `why`, so existing readers of that field are unaffected.
FLAG_PROSE = {
    "control_sample": "control sample: this verdict measures the auto-accept gate",
    "mask_box": "mask != box",
    "class_conflict": "class conflict",
    "rare_class": "rare class",
    "low_conf": "low conf",
    "review_band": "review band",
    "settlement_sample": "settlement sample: this verdict decides whether the lot settles",
    "settlement_spot": "settlement spot check: this verdict audits a settled lot",
}

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _why_and_priority(obj: Object, onto) -> tuple[str, float, list[dict]]:
    """The reasons this object is in the queue, its rank, and those reasons as sorted structured flags."""
    c = onto.by_id(obj.class_id)
    prov = obj.provenance or {}
    rare = c.india or c.l1 == "fallback"
    mask_box = mask_box_disagrees(prov)
    conflict = class_conflict(prov)

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

    # The rank itself lives in services/queue_order.py, beside its SQL twin and the gold measurement behind it.
    return why, priority_of(obj.conf, obj.class_id, prov, rare_class_ids(onto)), flags


@router.get("/triage", response_model=list[TriageRow])
async def triage(
    db: AsyncSession = Depends(db_session),
    states: str = Query("review,annotate"),
    session_id: str | None = None,
    klass: str | None = None,
    city: str | None = None,
    flywheel: str | None = None,
    control: bool = False,
    limit: int = 200,
    hidden=Depends(caller_hidden_projects),
):
    limit = min(max(limit, 1), 1000)
    onto = get_ontology()
    state_list = [s.strip() for s in states.split(",") if s.strip()]
    stmt = (
        select(Object, Frame.session_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .join(DbSession, Frame.session_id == DbSession.session_id)
    )
    stmt = scope_visible_sessions(stmt, hidden)
    if control:
        # The control queue: objects the gate auto-accepted that were randomly sampled for an always-human
        # verdict. The sample defines the queue, so the states filter does not apply - these objects sit in
        # auto_accept, which the default filter would exclude, and excluding them is how the corpus reached
        # 601 sampled / 0 judged. Each verdict here moves the gate's measured precision (the number the
        # 0.97 drift floor watches), which is why they outrank ordinary review work.
        from db.models import ControlSample

        stmt = stmt.join(ControlSample, ControlSample.object_id == Object.object_id).where(
            ControlSample.human_verdict.is_(None))
    else:
        stmt = stmt.where(Object.state.in_(state_list))
    if session_id:
        stmt = stmt.where(Frame.session_id == UUID(session_id))
    if city:
        stmt = stmt.where(DbSession.city == city)
    if klass:
        stmt = stmt.where(Object.class_id == onto.by_name(klass).id)
    draw_order: dict[str, int] | None = None
    if flywheel:
        # the worklist a flywheel cycle dispatched: objects it stamped with this cycle id
        stmt = stmt.where(Object.provenance["flywheel"]["cycle_id"].astext == flywheel)
        if flywheel.startswith("settle-"):
            # A settlement sample is a random draw and its verdicts tally in increments. Served
            # hardest-first, the judged prefix would be the lot's worst crops, not a sample of it;
            # served in draw order, every completed prefix is one. The uncertainty formula stays as it
            # is for every other worklist.
            from db.models import SettlementLot

            lot = (await db.execute(select(SettlementLot).where(
                SettlementLot.batch_id == flywheel))).scalars().first()
            if lot is not None:
                draw_order = {oid: i for i, oid in enumerate(lot.sample_object_ids or [])}
    if draw_order is None:
        # Ranked in the database across every matching object, then the page taken. Over-fetching an
        # unordered 3x and ranking that served the top of whatever Postgres happened to read first.
        stmt = stmt.order_by(priority_sql(rare_class_ids(onto)).desc(), Object.object_id).limit(limit)

    rows = (await db.execute(stmt)).all()
    out: list[TriageRow] = []
    for obj, sid in rows:
        why, priority, flags = _why_and_priority(obj, onto)
        if draw_order is not None:
            idx = draw_order.get(str(obj.object_id), len(draw_order))
            priority = round(1.0 + (len(draw_order) - idx) / max(1, len(draw_order)), 4)
            flags = [{"code": "settlement_sample", "label": FLAG_LABEL["settlement_sample"],
                      "severity": FLAG_SEVERITY["settlement_sample"]}, *flags]
            why = f"{FLAG_PROSE['settlement_sample']} (draw {idx + 1} of {len(draw_order)}), {why}"
        elif flywheel and flywheel.startswith("spot-"):
            flags = [{"code": "settlement_spot", "label": FLAG_LABEL["settlement_spot"],
                      "severity": FLAG_SEVERITY["settlement_spot"]}, *flags]
            why = f"{FLAG_PROSE['settlement_spot']}, {why}"
        if control:
            flags = [{"code": "control_sample", "label": FLAG_LABEL["control_sample"],
                      "severity": FLAG_SEVERITY["control_sample"]}, *flags]
            why = f"{FLAG_PROSE['control_sample']}, {why}"
            priority = round(priority + 1.0, 4)
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


@router.get("/triage/ordering")
async def triage_ordering(gold_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Why the queue is in this order: each candidate ordering, measured on a gold set by how many of the
    wrong labels a reviewer meets in the first 10, 25 and 50% of the queue (services/queue_order.py).
    Without a gold id, the newest gold set. Recomputed on each call; it takes a few seconds."""
    from fastapi import HTTPException

    from db.models import GoldSet
    from services.queue_order import evaluate_orderings

    if gold_id is None:
        gold_id = (await db.execute(select(GoldSet.gold_id).order_by(GoldSet.created_at.desc()).limit(1))).scalar()
        if gold_id is None:
            raise HTTPException(404, "no gold set yet: seal one in Evaluation to measure the queue order")
    try:
        return await evaluate_orderings(db, gold_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
