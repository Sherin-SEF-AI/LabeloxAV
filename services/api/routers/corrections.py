"""Interactive AI correction endpoints: suggest similar objects to bulk-fix after a correction, the
confusion view (what the model gets wrong, from the Review audit trail), and embedding coverage."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.observability import spawn
from db.models import Frame, Object, ObjectEmbedding, Review
from db.models import Session as DbSession
from services.api.deps import db_session
from services.autolabel.ontology import get_ontology
from services.intelligence.corrections import correction_candidates
from services.training.gpu_lease import training_holds_gpu

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
    # Off by default: one systematic error is usually spread over several source classes, and scoping to the
    # class the operator happened to start from surfaces one lineage of several.
    same_class: bool = False


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
        same_class=body.same_class,
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
    """How much of the corpus similar-search can actually reach.

    This counted the legacy `Embedding` table, which nothing has written since the move to pgvector, so it
    reported near zero against a corpus that is in fact fully embedded. The live tables are
    `object_embedding` and `frame_embedding`.

    Reported by vector rather than by row, because a row is not a vector: `siglip_vec` arrived later as a
    nullable column, and a crop holding a DINOv3 vector and a NULL SigLIP2 one is reachable by
    find-similar and unreachable by text. One number cannot say both, so it does not try.
    """
    total = (await db.execute(select(func.count()).select_from(Object).where(Object.state != "rejected"))).scalar_one()
    dino = (await db.execute(select(func.count()).select_from(ObjectEmbedding)
                             .where(ObjectEmbedding.dino_vec.isnot(None)))).scalar_one()
    siglip = (await db.execute(select(func.count()).select_from(ObjectEmbedding)
                               .where(ObjectEmbedding.siglip_vec.isnot(None)))).scalar_one()

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    return {"total": int(total),
            # `embedded` and `pct` keep their meaning for existing callers: what find-similar can reach.
            "embedded": int(dino), "pct": pct(dino),
            "visual_embedded": int(dino), "visual_pct": pct(dino),
            "text_embedded": int(siglip), "text_pct": pct(siglip)}


@router.post("/corrections/embed")
async def embed(session_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Compute CLIP object embeddings (a session, or the whole corpus) in the background so the
    similar-search has coverage. GPU work; yields to a running training job."""
    if await training_holds_gpu(db):
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

    spawn(_run(), name="_run")
    return {"started": True}
