"""Rare-scenario discovery (M1.5). Two complementary signals over a session:

  embedding novelty: cluster DINOv3 frame vectors (HDBSCAN); frames that are noise or far from the
                     corpus centroid become embedding_outlier candidates, members of the smallest
                     clusters become sparse_cluster candidates.
  rare-class:        frames containing low-frequency classes (the india / animal / fallback set, reusing
                     gate.is_rare) become rare_class candidates.

Writes scenario_candidate rows for a human confirm/dismiss/tag queue. This is the embedding-plus-heuristic
version that finds unusual frames now; full trajectory-based scenario mining stays in the miner. Feeds
active learning and sellable rare slices.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from uuid import UUID

import numpy as np
from sqlalchemy import delete, select

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, FrameEmbedding, Object, ScenarioCandidate
from db.session import get_sessionmaker
from services.autolabel.gate import is_rare
from services.autolabel.ontology import get_ontology

log = get_logger("discovery")


async def discover_session(session_id: UUID, max_per_kind: int = 60) -> dict:
    from hdbscan import HDBSCAN

    cfg = get_settings().intel.discovery
    onto = get_ontology()
    maker = get_sessionmaker()

    async with maker() as db:
        rows = (await db.execute(
            select(Frame.frame_id, FrameEmbedding.dino_vec)
            .join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, FrameEmbedding.dino_vec.isnot(None)))).all()

    cands: list[tuple] = []  # (frame_id, kind, score, cluster_id, rare_classes)
    if len(rows) >= max(cfg.min_cluster_size, 5):
        ids = [r[0] for r in rows]
        mat = np.asarray([r[1] for r in rows], dtype=np.float32)
        labels = HDBSCAN(min_cluster_size=cfg.min_cluster_size, metric="euclidean").fit_predict(mat)
        centroid = mat.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        dist = 1.0 - mat @ centroid  # distance to corpus centroid (vectors are normalized)
        thresh = float(np.quantile(dist, cfg.outlier_quantile))
        sizes = Counter(labels)
        clustered_sizes = [s for lbl, s in sizes.items() if lbl != -1]
        small_cut = np.median(clustered_sizes) / 2 if clustered_sizes else 0

        for i, fid in enumerate(ids):
            lbl = int(labels[i])
            if lbl == -1 or dist[i] >= thresh:
                cands.append((fid, "embedding_outlier", float(dist[i]), None if lbl == -1 else lbl, None))
            elif sizes[lbl] <= small_cut:
                cands.append((fid, "sparse_cluster", float(1.0 / sizes[lbl]), lbl, None))

    # rare-class signal
    async with maker() as db:
        obj_rows = (await db.execute(
            select(Object.frame_id, Object.class_id).join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == session_id, Object.state != "rejected"))).all()
    frame_rares: dict = defaultdict(set)
    for fid, cid in obj_rows:
        if is_rare(cid, onto):
            frame_rares[fid].add(onto.by_id(cid).name)
    for fid, names in frame_rares.items():
        cands.append((fid, "rare_class", float(len(names)), None, sorted(names)))

    # rank + cap per kind, then persist (replace this session's pending candidates)
    by_kind: dict = defaultdict(list)
    for c in cands:
        by_kind[c[1]].append(c)
    kept: list[tuple] = []
    for kind, items in by_kind.items():
        kept.extend(sorted(items, key=lambda c: -c[2])[:max_per_kind])

    async with maker() as db:
        await db.execute(delete(ScenarioCandidate).where(
            ScenarioCandidate.session_id == session_id, ScenarioCandidate.state == "pending"))
        for fid, kind, score, cluster_id, rares in kept:
            db.add(ScenarioCandidate(session_id=session_id, frame_id=fid, kind=kind, score=round(score, 4),
                                     cluster_id=cluster_id, rare_classes=rares, state="pending"))
        await db.commit()

    counts = Counter(k for _, k, *_ in kept)
    out = {"session_id": str(session_id), "candidates": len(kept), "by_kind": dict(counts)}
    log.info("discovery.done", **out)
    return out
