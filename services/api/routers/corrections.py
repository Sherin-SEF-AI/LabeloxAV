"""Interactive AI correction endpoints: suggest similar objects to bulk-fix after a correction, the
confusion view (what the model gets wrong, from the Review audit trail), and embedding coverage."""

from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Embedding, Frame, Object, Review, TrainingJob
from db.models import Session as DbSession
from services.api.deps import db_session
from services.autolabel.ontology import get_ontology
from services.intelligence.corrections import correction_candidates

log = get_logger("api_corrections")
router = APIRouter()


class SuggestIn(BaseModel):
    object_id: str
    kind: str = "class"               # class | attr
    old_class_name: str | None = None  # class kind: the wrong class to search for
    new_class_name: str | None = None  # class kind: the corrected class
    attr_key: str | None = None        # attr kind
    old_value: object | None = None    # attr kind: the wrong value
    new_value: object | None = None    # attr kind: the corrected value
    filters: dict = {}                 # cam_id, city, conf_min/max, area_min/max
    limit: int = 200
    threshold: float = 0.82


@router.post("/corrections/suggest")
async def suggest(body: SuggestIn, db: AsyncSession = Depends(db_session)):
    onto = get_ontology()
    old_class_id = None
    if body.kind == "class":
        if not body.old_class_name or not onto.has_name(body.old_class_name):
            raise HTTPException(400, "old_class_name required and must be a known class")
        old_class_id = onto.by_name(body.old_class_name).id
        change = {"old": body.old_class_name, "new": body.new_class_name}
        new_value = body.new_class_name
    else:
        if not body.attr_key:
            raise HTTPException(400, "attr_key required for kind='attr'")
        change = {"attr": body.attr_key, "old": body.old_value, "new": body.new_value}
        new_value = body.new_value

    res = await correction_candidates(
        db, body.object_id, kind=body.kind, old_class_id=old_class_id,
        attr_key=body.attr_key, old_value=body.old_value, new_value=new_value,
        filters=body.filters, limit=body.limit, threshold=body.threshold,
    )
    return {"kind": body.kind, "change": change, **res}


@router.get("/corrections/confusions")
async def confusions(by: str = "class", limit: int = 30, db: AsyncSession = Depends(db_session)):
    """Aggregate the Review audit trail into confusion pairs (old class -> corrected class x count).
    `by=camera|city` additionally groups by that dimension. The learn-from-corrections signal."""
    onto = get_ontology()
    if by == "camera":
        stmt = (select(Review.before, Review.after, Frame.cam_id)
                .join(Object, Review.object_id == Object.object_id)
                .join(Frame, Object.frame_id == Frame.frame_id))
    elif by == "city":
        stmt = (select(Review.before, Review.after, DbSession.city)
                .join(Object, Review.object_id == Object.object_id)
                .join(Frame, Object.frame_id == Frame.frame_id)
                .join(DbSession, Frame.session_id == DbSession.session_id))
    else:
        stmt = select(Review.before, Review.after)
    rows = (await db.execute(stmt)).all()

    c: Counter = Counter()
    for r in rows:
        before, after = r[0] or {}, r[1] or {}
        dim = r[2] if len(r) > 2 else None
        b, a = before.get("class_id"), after.get("class_id")
        if b is None or a is None or b == a:
            continue
        c[(b, a, dim)] += 1

    out = []
    for (b, a, dim), n in c.most_common(limit):
        try:
            row = {"old_class": onto.by_id(b).name, "new_class": onto.by_id(a).name, "count": n}
        except Exception:  # noqa: BLE001
            continue
        if dim is not None:
            row["group"] = dim
        out.append(row)
    return {"by": by, "total_corrections": sum(c.values()), "confusions": out}


@router.get("/corrections/coverage")
async def coverage(db: AsyncSession = Depends(db_session)):
    total = (await db.execute(select(func.count()).select_from(Object).where(Object.state != "rejected"))).scalar_one()
    emb = (await db.execute(select(func.count()).select_from(Embedding))).scalar_one()
    return {"embedded": int(emb), "total": int(total), "pct": round(100 * emb / total, 1) if total else 0.0}


@router.post("/corrections/embed")
async def embed(session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Compute CLIP object embeddings (a session, or the whole corpus) in the background so the
    similar-search has coverage. GPU work; yields to a running training job."""
    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
        raise HTTPException(503, "GPU reserved for a training job; embedding is paused until it finishes")

    async def _run() -> None:
        from uuid import UUID as _UUID

        from db.session import get_sessionmaker
        from services.intelligence.embeddings import compute_session_embeddings

        try:
            if session_id:
                await compute_session_embeddings(_UUID(session_id))
            else:
                async with get_sessionmaker()() as d:
                    sids = (await d.execute(select(DbSession.session_id))).scalars().all()
                for sid in sids:
                    await compute_session_embeddings(sid)
        except Exception as exc:  # noqa: BLE001
            log.error("corrections.embed_failed", error=str(exc))

    asyncio.create_task(_run())
    return {"started": True}
