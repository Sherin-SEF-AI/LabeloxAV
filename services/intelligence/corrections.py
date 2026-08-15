"""Interactive AI correction: given ONE human correction, find the objects that share the mistake.

This searched a table nothing writes to, and so had never returned a result. It read the legacy CLIP
`embedding` table while the pipeline moved to pgvector `object_embedding` two commits earlier; every other
find-similar surface was migrated and this one file was missed, including the coverage endpoint of this very
feature (`services/api/routers/corrections.py`), which reads the new table and reported healthy coverage
while the search read an empty one. Measured on the live corpus: 39 rows in the table it queried against
567,527 in the table it needed, and 0 of those 39 in the class being corrected.

Two consequences worth naming, because they are why nobody caught it. The old `_source_vector` embedded the
query object on demand and wrote the result back into the dead table, so the table slowly filled with exactly
the objects that had just been corrected, which then carried the NEW class and could never match the OLD
class filter. And the empty result was reported as "no similar objects above the threshold", which is a
statement about similarity, when the truth was that there was nothing to compare against at all.

The candidate set is now visually similar objects REGARDLESS of current class. The relabel agent spread one
mistake across three source classes (1,047 from `bus`, 708 from `traffic_sign`, 522 from `hoarding`, all into
`bmtc_bus_shelter`), so a correction scoped to the class the operator happened to start from would surface
one lineage of three. The class of every candidate is returned so a person can see what they are agreeing to,
and `same_class` remains available for corrections that really are class-specific.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import tune_recall
from core.logging import get_logger
from db.models import Frame, Object, ObjectEmbedding
from db.models import Session as DbSession
from services.autolabel.ontology import get_ontology

log = get_logger("corrections")

# Reasons an empty result is not "nothing looks like this". Returned to the client and rendered, because the
# three are indistinguishable to a person and only one of them is about similarity.
NOT_EMBEDDED = "this object has no visual embedding yet, so nothing can be compared against it"
NO_NEIGHBOURS = "no objects above the similarity threshold"

# Fetched before thresholding, so lowering the slider does not need another round trip and a tight cluster
# does not crowd out everything else.
OVER_FETCH = 4


async def _source_vector(db: AsyncSession, object_id: UUID):
    """The corrected object's DINOv3 vector, or None.

    Deliberately does not embed on demand. The old version did, into the wrong table and with the wrong
    model, and a request handler is the wrong place to load a vision model anyway: the embedding daemon
    (`services/intelligence/embed/daemon.py`) covers new objects incrementally, and an object it has not
    reached yet is a fact the caller can be told rather than a stall.
    """
    emb = await db.get(ObjectEmbedding, object_id)
    return emb.dino_vec if emb is not None and emb.dino_vec is not None else None


def _apply_metadata_filters(stmt, filters: dict, *, attr_key=None, old_value=None):
    """The narrowing a person asked for, in SQL rather than after the fact.

    These have to be part of the indexed query rather than a post-filter over the top-k: filtering after the
    ANN returns its window is how a camera filter turns twenty candidates into zero while a hundred perfectly
    good ones sit just outside the window.
    """
    if attr_key is not None:
        stmt = stmt.where(Object.attrs[attr_key].astext == str(old_value))

    cam, city = filters.get("cam_id"), filters.get("city")
    if cam or city:
        stmt = stmt.join(Frame, Object.frame_id == Frame.frame_id)
        if cam:
            stmt = stmt.where(Frame.cam_id == cam)
        if city:
            stmt = stmt.join(DbSession, Frame.session_id == DbSession.session_id).where(DbSession.city == city)
    if filters.get("conf_min") is not None:
        stmt = stmt.where(Object.conf >= filters["conf_min"])
    if filters.get("conf_max") is not None:
        stmt = stmt.where(Object.conf <= filters["conf_max"])
    # bbox is xyxy; Postgres arrays are 1-based, so area = (x2-x1)*(y2-y1) reads as [3]-[1] by [4]-[2].
    area = (Object.bbox[3] - Object.bbox[1]) * (Object.bbox[4] - Object.bbox[2])
    if filters.get("area_min") is not None:
        stmt = stmt.where(area >= filters["area_min"])
    if filters.get("area_max") is not None:
        stmt = stmt.where(area <= filters["area_max"])
    return stmt


async def correction_candidates(
    db: AsyncSession, object_id: str, *, kind: str, old_class_id=None, attr_key=None,
    old_value=None, new_value=None, filters: dict | None = None, limit: int = 200,
    threshold: float = 0.82, same_class: bool = False,
) -> dict:
    """Objects that look like the one just corrected, richest match first.

    `same_class` scopes to the class being corrected away from. Off by default: one systematic error is
    usually spread over several source classes, and scoping to one of them hides the rest.
    """
    filters = filters or {}
    oid = UUID(object_id)
    qvec = await _source_vector(db, oid)
    if qvec is None:
        return {"count": 0, "candidates": [], "reason": NOT_EMBEDDED}

    await tune_recall(db)
    dist = ObjectEmbedding.dino_vec.cosine_distance(list(map(float, qvec))).label("d")
    stmt = (select(ObjectEmbedding.object_id, dist, Object.class_id, Object.conf, Object.state,
                   Object.frame_id, Object.attrs, Object.source, Object.track_id)
            .join(Object, Object.object_id == ObjectEmbedding.object_id)
            .where(ObjectEmbedding.object_id != oid, Object.state != "rejected"))
    if same_class and old_class_id is not None:
        stmt = stmt.where(Object.class_id == old_class_id)
    stmt = _apply_metadata_filters(stmt, filters, attr_key=attr_key, old_value=old_value)

    rows = (await db.execute(stmt.order_by(dist).limit(min(limit, 500) * OVER_FETCH))).all()

    onto = get_ontology()
    out: list[dict] = []
    for cand_id, d, class_id, conf, state, frame_id, attrs, source, track_id in rows:
        sim = 1.0 - float(d)
        if sim < threshold:
            continue
        try:
            class_name = onto.by_id(class_id).name
        except KeyError:
            # A class the ontology no longer knows is still a real object on somebody's screen; naming it by
            # id beats dropping it from the list without saying so.
            class_name = f"class {class_id}"
        current = class_name if kind == "class" else (attrs or {}).get(attr_key)
        out.append({
            "object_id": str(cand_id), "frame_id": str(frame_id),
            "class_name": class_name, "current": current,
            "conf": float(conf) if conf is not None else None, "state": state,
            "score": round(sim, 4), "source": source,
            # One mislabelled object appears once per frame of its track. Ninety near-identical crops of the
            # same billboard is a legitimate result, since every one of them is wrong and needs fixing, but
            # it reads as ninety separate findings unless the grouping is said out loud.
            "track_id": str(track_id) if track_id else None,
            "crop_url": f"/api/objects/{cand_id}/crop",
            # Already at the corrected value, so it is shown but not pre-selected.
            "already": current == new_value,
            # A human already ruled on this one. Applying a batch over it would overwrite somebody's
            # decision with a guess, which is the one thing bulk tooling must not do quietly.
            "human": source == "human",
        })
        if len(out) >= limit:
            break

    log.info("corrections.candidates", object_id=object_id, kind=kind, examined=len(rows),
             returned=len(out), threshold=threshold, same_class=same_class)
    tracks = {c["track_id"] for c in out if c["track_id"]}
    return {"count": len(out), "candidates": out,
            "reason": None if out else NO_NEIGHBOURS,
            "examined": len(rows),
            # How many distinct objects the count actually represents, as opposed to how many rows.
            "n_tracks": len(tracks),
            "untracked": sum(1 for c in out if not c["track_id"])}
