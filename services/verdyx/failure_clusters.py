"""What a model's failures look like, as groups rather than as a list.

`Evaluation.failure_clusters` has a documented shape and has always been written `{}`, unconditionally, at
the one place an evaluation is recorded. So `services/sievyx/failure_mining.py:mine_from_failures`, which
ranks the unlabeled pool by similarity to a failure mode, has never had an input and has never had a caller.
Both ends were built and the middle was missing.

The middle is small, because everything it needs already exists. `EvalPatch` records every prediction the
harness scored as a true positive, false positive or false negative, against a sealed gold set and one
immutable inference run. A false negative carries the `object_id` of the ground truth the model failed to
find, and that object has a DINOv3 vector. Grouping those vectors says what the misses have in common, and
that is precisely the query "find me more things like what the model cannot see".

Only false negatives are clustered, and the reason is not a preference. A false positive is a prediction, so
it has no `Object` row and therefore no embedding; embedding one means cropping its box and running the
encoder, which is real GPU work and a larger change. The stored result says so explicitly rather than
letting a caller assume the clusters cover both, because a failure map that silently omits half the failure
modes is worse than one that names what it left out.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EvalPatch, ObjectEmbedding, OntologyClass

log = get_logger("verdyx.failure_clusters")

# Misses are rare by construction on a small gold set: the live corpus holds 108 false negatives across
# every evaluation ever run. HDBSCAN's default minimum of 15 would return nothing but noise on a set that
# size, so the floor is set to the smallest group that is still a pattern rather than a coincidence.
MIN_CLUSTER_SIZE = 3

# How much of a cluster must share a ground-truth class before the cluster is described by that class.
# A majority, because below it the group is held together by appearance rather than by label, and calling
# it "missed motorcycle" when one member of three is a motorcycle is a confident wrong sentence.
MIN_CLASS_PURITY = 0.5


async def resolve_patch_eval_id(db: AsyncSession, *, gold_id: str | None,
                                model_version: str | None) -> UUID | None:
    """The patch set belonging to a VERDYX evaluation of this model against this gold set.

    `EvalPatch.eval_id` is not `Evaluation.eval_id`. There is no foreign key between them and, on the live
    corpus, the two id sets do not intersect at all: patches are written by the analytics evaluator, which
    mints its own id. What the two tables genuinely share is `(gold_id, model_version)`, so that is the join.

    Getting this wrong is silent rather than loud. Passing a VERDYX eval id straight into a patch query
    matches nothing and produces an empty, perfectly well-formed cluster payload.
    """
    if not gold_id or not model_version:
        return None
    return (await db.execute(
        select(EvalPatch.eval_id)
        .where(EvalPatch.gold_id == gold_id, EvalPatch.model_version == model_version)
        .order_by(EvalPatch.created_at.desc()).limit(1))).scalar_one_or_none()


async def build_failure_clusters(db: AsyncSession, patch_eval_id: UUID | str | None,
                                 *, min_cluster_size: int = MIN_CLUSTER_SIZE) -> dict:
    """Group an evaluation's misses by appearance. Returns the `failure_clusters` payload.

    Takes the id of a *patch* set (see `resolve_patch_eval_id`), not a VERDYX evaluation id.

    Keyed by cluster id, each entry carrying the member object ids, the size, and a `condition` naming what
    the group has in common, which is what makes a cluster actionable rather than a bag of uuids.

    Member ids rather than a stored centroid: a 768-dimensional vector per cluster in JSONB is heavy, and it
    would be a copy that can go stale against the embeddings it was derived from. The miner recomputes
    centroids from these ids, so there is one source of truth.
    """
    from services.curation.projection import cluster_labels

    if patch_eval_id is None:
        return {"clusters": {}, "coverage": {},
                "reason": "no scored patch set for this model and gold set, so there are no misses to group"}
    eid = patch_eval_id if isinstance(patch_eval_id, UUID) else UUID(str(patch_eval_id))

    rows = (await db.execute(
        select(EvalPatch.object_id, EvalPatch.gt_class_id, ObjectEmbedding.dino_vec, OntologyClass.name)
        .join(ObjectEmbedding, ObjectEmbedding.object_id == EvalPatch.object_id)
        .outerjoin(OntologyClass, OntologyClass.id == EvalPatch.gt_class_id)
        .where(EvalPatch.eval_id == eid, EvalPatch.outcome == "fn",
               ObjectEmbedding.dino_vec.isnot(None)))).all()

    # Counted before clustering so the payload can say how much of the failure set it actually describes.
    fp_total = (await db.execute(
        select(EvalPatch.patch_id).where(EvalPatch.eval_id == eid, EvalPatch.outcome == "fp"))).all()
    fn_total = (await db.execute(
        select(EvalPatch.patch_id).where(EvalPatch.eval_id == eid, EvalPatch.outcome == "fn"))).all()

    covers = {"false_negatives_clustered": len(rows), "false_negatives_total": len(fn_total),
              "false_positives_total": len(fp_total),
              "false_positives_clustered": 0,
              "note": ("false positives are predictions and carry no Object row, so they have no embedding "
                       "to cluster; clustering them needs their boxes cropped and encoded")}

    if len(rows) < min_cluster_size:
        log.info("verdyx.failure_clusters.too_few", eval_id=str(eid), n=len(rows))
        return {"clusters": {}, "coverage": covers,
                "reason": f"only {len(rows)} embeddable misses, fewer than the {min_cluster_size} "
                          "needed for a group to mean anything"}

    X = np.asarray([list(r[2]) for r in rows], dtype=np.float32)
    labels = cluster_labels(X, min_cluster_size=min_cluster_size)

    clusters: dict[str, dict] = {}
    for cid in sorted({int(v) for v in labels if int(v) >= 0}):
        members = [i for i, v in enumerate(labels) if int(v) == cid]
        names = [rows[i][3] for i in members if rows[i][3]]
        # The dominant ground-truth class is the most useful one-line description of a group of misses:
        # "the model keeps failing to find cattle" is a condition somebody can act on.
        top = max(set(names), key=names.count) if names else None
        share = (names.count(top) / len(names)) if names and top else 0.0
        # Name a class only when the group is actually about that class. A three-member cluster holding
        # three different classes is a group of similar-looking misses, not "missed motorcycle", and
        # labelling it so would put a confident wrong sentence in front of whoever reads the failure map.
        if top and share >= MIN_CLASS_PURITY:
            condition = f"missed {top} ({names.count(top)} of {len(members)})"
        elif names:
            condition = ("visually similar misses across "
                         f"{len(set(names))} classes ({', '.join(sorted(set(names)))})")
        else:
            condition = f"missed objects ({len(members)})"
        clusters[str(cid)] = {
            "condition": condition,
            "dominant_class": top if share >= MIN_CLASS_PURITY else None,
            "class_purity": round(share, 3),
            "member_object_ids": [str(rows[i][0]) for i in members],
            "size": len(members),
        }

    noise = int((labels < 0).sum())
    log.info("verdyx.failure_clusters.built", eval_id=str(eid), clusters=len(clusters),
             clustered=len(rows) - noise, noise=noise)
    out = {"clusters": clusters, "coverage": {**covers, "unclustered_as_noise": noise}}
    if not clusters:
        # An empty result has two very different meanings and the caller cannot tell them apart from `{}`.
        # This one is "there were misses and none of them resembled each other", which on a gold set this
        # small is the ordinary outcome, not a failure to run.
        out["reason"] = (f"{len(rows)} misses clustered to nothing: no {min_cluster_size} of them were "
                         "close enough in appearance to form a group")
    return out


def centroids_from_clusters(payload: dict, vectors_by_object_id: dict) -> tuple[list, list[str]]:
    """Centroids for `mine_from_failures`, recomputed from member ids rather than read from storage.

    Returns (centroids, cluster_ids) aligned, so a mined result's `failure_mode` index maps back to a named
    cluster instead of an anonymous integer.
    """
    centroids, ids = [], []
    for cid, c in (payload.get("clusters") or {}).items():
        vecs = [vectors_by_object_id[o] for o in c.get("member_object_ids", [])
                if o in vectors_by_object_id]
        if not vecs:
            continue
        centroids.append(np.mean(np.asarray(vecs, dtype=np.float64), axis=0))
        ids.append(cid)
    return centroids, ids
